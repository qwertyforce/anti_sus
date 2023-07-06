import datetime as dt
from PIL import Image
from requests import get, post, head
import os
import re
import io 
import json
from tqdm import tqdm
from time import sleep
import numpy as  np
import concurrent.futures
import traceback       
import flickrapi
import zmq

flickr_api_key = u''
flickr_api_secret = u''
flickr = flickrapi.FlickrAPI(flickr_api_key, flickr_api_secret, format='parsed-json')

IMG_DOWNLOAD_MAX_WORKERS = 32
IMG_PATH = "./test/"
JSON_PATH  = "./test_json/"
EXTENSION={"image/jpg":".jpg","image/jpeg":".jpg","image/png":".png"}
ALLOWED_MIME=["image/jpg","image/jpeg", "image/png"]
IMGUR_CLIENT_ID=""
POST_TO_SCENERY = False
IMPORT_IMAGES_BOT_PASSWORD = "123"
REQUEST_TIMEOUT = 120000 # ms   2 minutes
REQUEST_RETRIES = 5

context = zmq.Context()
print("Connecting to anti_sus server…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:7777")

try:
    os.mkdir(IMG_PATH)
    os.mkdir(JSON_PATH)
except:
    print("folders exist")

def b58decode(s):
    alphabet = '123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'
    num = len(s)
    decoded = 0 
    multi = 1
    for i in reversed(range(0, num)):
        decoded = decoded + multi * ( alphabet.index( s[i] ) )
        multi = multi * len(alphabet)
    return decoded
    
def get_mime(url):
    try:
        x = head(url,timeout=5)
        if "content-type" in x.headers:
            return x.headers["content-type"]
        if "Content-Type" in x.headers:
            return x.headers["Content-Type"]
        return False
    except:
        return False

def check_fit(images):
    global socket
    retries_left = REQUEST_RETRIES
    socket.send(images,copy=False)
    print("sent main")
    while True:
        if (socket.poll(REQUEST_TIMEOUT) & zmq.POLLIN) != 0:
            print("waiting for answer")
            message = socket.recv()
            return list(np.frombuffer(message,dtype=np.int32))
            # else:
            #     logging.error("Malformed reply from server: %s", reply)
            #     continue

        retries_left -= 1
        print("No response from server")
        # Socket is confused. Close and remove it.
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()

        print(f"Reconnecting to server... retries_left: {retries_left}")
        # Create new connection
        socket = context.socket(zmq.REQ)
        socket.connect("tcp://localhost:7777")
        if retries_left == 0:
            print("Server seems to be offline, abandoning")
            # sys.exit()
            return []
        print("Resending images")
        socket.send(images,copy=False)

def download(url, file_name, ext, post_idx):
    try:
        if ext is None:
            file_mime=get_mime(url)
            if file_mime in ALLOWED_MIME:
                ext=EXTENSION[file_mime]
            else:
                return
        full_file_name=IMG_PATH+file_name+ext
        if os.path.isfile(full_file_name):
            # print("File exist")
            return False
        
        if "imgur.com" in url: # if ip banned by imgur, use this
            url="https://proxy.duckduckgo.com/iu/?u="+url

        response = get(url,timeout=5, headers={"User-Agent": """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"""})

        fake_file = io.BytesIO(response.content)
        
        im = Image.open(fake_file)
        width, height = im.size
        if (width*height < 1280 * 1024) or (width/height > 3) or (height/width > 2):
            return False
        if im.mode != 'RGB':
            im = im.convert('RGB')
        return (post_idx, np.array(im.resize((512,512),Image.Resampling.LANCZOS)), full_file_name, response.content)
    except Exception as e:
        traceback.print_exc()
        print("error "+ url)

def handle_imgur(post, imgur_link):
    try:
        album_id=re.search('(?<=\/a\/)(.*)', imgur_link).group(0)
        if album_id:
            print(f"downloading album {imgur_link}")
            url = f"https://api.imgur.com/3/album/{album_id}/images"
            headers = {'Authorization': f'Client-ID {IMGUR_CLIENT_ID}'}
            response = get(url, headers=headers,timeout=5)
            data = response.json()
            post__img_urls = []
            for img in data["data"]:
                if img["type"] in ALLOWED_MIME:
                   post__img_urls.append((post, img["link"], post["id"]+"_imgur_"+img["id"], EXTENSION[img["type"]]))

            return post__img_urls
    except Exception as e:
        traceback.print_exc()
        print("error "+ imgur_link)
        return []

def flick_get_best_photo(photo_id):
    sizes = flickr.photos_getSizes(photo_id=photo_id)
    img = sizes["sizes"]["size"][-1]  # get best available version
    if img["media"] == "photo":
        return img["source"]
    return None

def get_post_img_url_post_id_ext(posts):
    post_img_url_post_id_ext=[]
    for post in tqdm(posts):
        if post["over_18"] or post["is_video"] or post["removed_by_category"]:
            continue
        if ("media_metadata" in post and post["media_metadata"]) or ("crosspost_parent_list" in post and post["crosspost_parent_list"] and "media_metadata" in post["crosspost_parent_list"][0] and post["crosspost_parent_list"][0]['media_metadata']):
            if "media_metadata" in post and post["media_metadata"]:
                media_metadata_obj = post["media_metadata"].values()
            elif post["crosspost_parent_list"][0]['media_metadata']:
                media_metadata_obj = post["crosspost_parent_list"][0]['media_metadata'].values()
            # else: #??????????????? how media_metadata_obj can be not assigned?
            #     continue
            for obj in media_metadata_obj:
                if "e" in obj and obj["e"] == "Image":
                    file_mime = obj["m"]
                    img_id = obj['id']
                    if file_mime in ALLOWED_MIME:
                        file_ext=EXTENSION[file_mime]
                        img_url = f"https://i.redd.it/{img_id}{file_ext}"
                        post_img_url_post_id_ext.append((post, img_url, f"{post['id']}_reddit_{img_id}", file_ext))
        else:
            if "imgur.com/a/" in post["url"] or "imgur.com/gallery" in post["url"]:
                post_img_url_post_id_ext.extend(handle_imgur(post,post["url"].replace("gallery","a")))

            elif "://imgur.com/" in post["url"] and not "." in post["url"]:
                img_url = post["url"] + ".jpg"
                post_img_url_post_id_ext.append((post,img_url, post["id"], None))
            
            elif "flic.kr/p/" in post["url"]:
                try:
                    start_id = post["url"].find("flic.kr/p/") + 10
                    end_id = post["url"].find("/",start_id)
                    if end_id == -1:
                        photo_id = b58decode(post["url"][start_id:])
                    else:
                        photo_id = b58decode(post["url"][start_id:end_id])
                    photo_id = str(photo_id)
                    img_url = flick_get_best_photo(photo_id)
                    if img_url:
                        post_img_url_post_id_ext.append((post,img_url, post["id"], None))
                except:
                    traceback.print_exc()
                    print("flic.kr/p/", post["url"])

            elif "flickr.com/photos/" in post["url"]:
                try:
                    start_id = post["url"].find("/",post["url"].index("/photos/")+8)+1
                    end_id = post["url"].find("/",start_id)
                    if end_id == -1:
                        photo_id = post["url"][start_id:]
                    else:
                        photo_id = post["url"][start_id:end_id]
                    img_url = flick_get_best_photo(photo_id)
                    if img_url:
                        post_img_url_post_id_ext.append((post,img_url, post["id"], None))
                except Exception as e:
                    traceback.print_exc()
                    print("flickr error ",post["url"])

            elif "staticflickr.com" in post["url"]:
                try:
                    start_id = post["url"].find(".jpg")
                    start_id = post["url"].rfind("_",0, post["url"].rfind("_",0,start_id))
                    end_id = post["url"].rfind("/",0,start_id)+1
                    id = post["url"][end_id:start_id]
                    if id.find("_") != -1:
                        id = id[:id.find("_")]
                    sizes = flickr.photos_getSizes(photo_id=id)
                    img = sizes["sizes"]["size"][-1]   # get best available version
                    if img["media"] == "photo":
                        img_url = img["source"]
                        post_img_url_post_id_ext.append((post, img_url, post["id"], None))
                except Exception as e: 
                    traceback.print_exc()
                    print("staticflickr error ",post["url"])
                    print("trying get flickr image directly",post["url"])
                    post_img_url_post_id_ext.append((post, post["url"], post["id"], None))
            else:
                post_img_url_post_id_ext.append((post, post["url"], post["id"], None))
    return post_img_url_post_id_ext

def clean_post_obj(post_obj):
    list_of_properties = ["selftext",
    "title",
    "media_metadata",
    "id",
    "author",
    "permalink",
    "url",
    "created_utc",
    "subreddit",
    "over_18",
    "is_video",
    "removed_by_category",
    "crosspost_parent_list"]
    new_obj={}
    for property in list_of_properties:
        if property in post_obj:
            new_obj[property] = post_obj[property]
    return new_obj
    
urls_broken = []
def scrape_reddit():
    # after_epoch=""
    empty_results=0
    while True:
        sleep(5)
        print('===iteration===')
        print(dt.datetime.now())
        data = get(f"https://www.reddit.com/r/all/new.json?sort=new&limit=100", headers={"User-Agent": """Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36"""})
        data=data.json()
       
        if len(data["data"]["children"]) == 0:
            print('len(data["data"]["children"]) == 0')
            sleep(20)
            empty_results+=1
        else:
            empty_results=0

        if empty_results==50:
            break
        
        posts = [clean_post_obj(obj["data"]) for obj in data["data"]["children"]]
        post_img_url_post_id_ext = get_post_img_url_post_id_ext(posts)

        post_idx_512img_filename_full_img= []
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=IMG_DOWNLOAD_MAX_WORKERS) as executor:
                future_to_url = (executor.submit(download, obj[1],obj[2],obj[3],post_idx) for post_idx, obj in enumerate(post_img_url_post_id_ext))
                for future in tqdm(concurrent.futures.as_completed(future_to_url,timeout=20)):
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(exc)
                    finally:
                        if data:
                            post_idx_512img_filename_full_img.append(data)            
        except:
            pass
         
        
        batch_size = 64
        for start_pos in tqdm(range(0,len(post_idx_512img_filename_full_img),batch_size)):
            batch = post_idx_512img_filename_full_img[start_pos:start_pos + batch_size]
            img_batch= np.array([x[1] for x in batch])
            print(img_batch.shape)
            check_fit_res = check_fit(img_batch)
            print(check_fit_res)
            for in_batch_idx in check_fit_res:
                post_idx = batch[in_batch_idx][0]
                post_data = post_img_url_post_id_ext[post_idx][0]
                source_url = "https://reddit.com" + post_data["permalink"]     #/r/.....
                print(source_url)
                with open(batch[in_batch_idx][2], "wb") as file:
                    if POST_TO_SCENERY:
                        post('http://127.0.0.1/import_image', files=dict(image=batch[in_batch_idx][3]), data=dict(source_url=source_url,tags='["from_nomad"]',import_images_bot_password=IMPORT_IMAGES_BOT_PASSWORD))
                    file.write(batch[in_batch_idx][3])

            uniq_post_idxs = set([batch[in_batch_idx][0] for in_batch_idx in check_fit_res])
            for post_idx in uniq_post_idxs:
                post_data = post_img_url_post_id_ext[post_idx][0]
                with open(JSON_PATH+post_data["id"]+".json", "w") as file:
                    file.write(json.dumps(post_data))
scrape_reddit()


