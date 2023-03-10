import zmq
import numpy as np
import torch
import clip
from torchvision import transforms
from PIL import Image

import pickle
import sklearn
with open("./gmm.model","rb") as file:
    gm = pickle.load(file)
GMM_THRESHOLD = 500

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using {device}")

model_wat = torch.jit.load('model_wat.pt')
model_wat.eval()
model_wat = model_wat.to(device)    

model, _ = clip.load("ViT-B/16", device=device, jit=True)
model.eval()
model = model.to(device)

_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def resize_to_224(images):
    new_images = torch.zeros((images.shape[0],3,224,224), dtype=torch.float32)
    for idx, image in enumerate(images):
        new_images[idx] = _transform(Image.fromarray(image).resize((224,224),Image.Resampling.LANCZOS))  # or just use bicubic everywhere, idk (torch, not pillow)
    return new_images

def get_features(images):
    with torch.no_grad():
        feature_vector = model.encode_image(images)
        feature_vector/=torch.linalg.norm(feature_vector,axis=1).reshape(-1,1)
    feature_vector = feature_vector.float()
    return feature_vector

def check_fit(images):
    features = get_features(images).cpu().numpy()
    scores = gm.score_samples(features)
    print(scores[np.where(scores > GMM_THRESHOLD)[0]])
    return np.where(scores > GMM_THRESHOLD)[0]
 
def check_watermarks(images):        # can make batch processing, but vram is low
    without_watermark = []
    for idx, image in enumerate(images):
        image = _transform(image).to(device)
        image = torch.stack([image[:,:224,:224], image[:,:224,224:], image[:,224:,:224], image[:,224:,224:]]) # quadrants
        features = get_features(image).reshape(1, 2048)
        with torch.no_grad():
            water_test = torch.sigmoid(model_wat(features)) 
            water_test = water_test.cpu().numpy()[0][0]
            if np.round(water_test)==0:
                without_watermark.append(idx)
    return without_watermark


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:7777")
print("starting...")
while True:
        message = socket.recv()
        images = np.frombuffer(message,dtype=np.uint8).reshape(-1,448, 448, 3)
        images_224 = resize_to_224(images).to(device)
        results_fit = check_fit(images_224)
        results_wat = check_watermarks(images[results_fit])
        final_results = results_fit[results_wat]
        # print(torch.cuda.memory_summary(device))
        socket.send(np.int32(final_results).tobytes())
