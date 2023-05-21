import zmq
import numpy as np
from PIL import Image
import pickle
import sklearn            #for pickled gmm
import onnxruntime as rt

with open("./gmm_16_r.model","rb") as file:
    gm = pickle.load(file)
GMM_THRESHOLD = 700

sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
# cpu_provider_options = {"arena_extend_strategy": "kSameAsRequested"}
cpu_provider_options={}
# sess_options.enable_cpu_mem_arena=False     #enable if problems with ram consumption
session_clip = rt.InferenceSession("./clip_visual.onnx", sess_options, providers=[("CPUExecutionProvider", cpu_provider_options)])
session_wat = rt.InferenceSession("./model_wat.onnx", sess_options, providers=[("CPUExecutionProvider", cpu_provider_options)])

def Normalize_np(mean,std):
    mean=np.array(mean)
    std=np.array(std)
    mean = mean.reshape(-1, 1, 1)
    std = std.reshape(-1, 1, 1)
    def normalize(images):
        images-=mean
        images/=std
        return images
    return normalize

def transform_bhwc_float(images):
    new_images = images.transpose(0, 3, 1, 2).astype(np.float32) #BHWC -> BCHW
    new_images/=255.0
    return new_images

normalize_wat = Normalize_np((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
normalize_clip = Normalize_np((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def sigmoid(x):
  return 1/(1 + np.exp(-x))

def resize_to_224(images):
    new_images = np.zeros((images.shape[0],224,224,3), dtype=np.float32)
    for idx, image in enumerate(images):
        new_images[idx] = np.array(Image.fromarray(image).resize((224,224),Image.Resampling.LANCZOS))
    return new_images

def get_features(images):                # batch size of 1, because onnx doesn't speedup on my cpu
    features=np.zeros((images.shape[0],512),dtype=np.float32)
    # print(images.shape)
    for idx, img in enumerate(images):
        img = img[np.newaxis,:]
        feature_vector = session_clip.run([], {'input':img})[0][0]
        features[idx]=feature_vector
    features/=np.linalg.norm(features,axis=1).reshape(-1,1)
    return features


# def get_features(images):                # batch inference
#     features = session_clip.run([], {'input':images})[0]
#     features/=np.linalg.norm(features,axis=1).reshape(-1,1)
#     return features

def check_fit(images):
    images_224 = resize_to_224(images)
    images_224 = transform_bhwc_float(images_224)
    images_224 = normalize_clip(images_224)
    
    features = get_features(images_224)
    scores = gm.score_samples(features)
    print(scores[np.where(scores > GMM_THRESHOLD)[0]])
    return np.where(scores > GMM_THRESHOLD)[0]
 
def check_watermarks(images):        # batch size of 1, because onnx doesn't speedup on my cpu
    images = transform_bhwc_float(images)
    images = normalize_wat(images)
    without_watermark = []
    for idx, img in enumerate(images):
        img = img[np.newaxis,:]
        output = session_wat.run([], {'input':img})[0][0][0]
        output = sigmoid(output)
        if np.round(output) == 0:
            without_watermark.append(idx)
    return np.array(without_watermark)

# def check_watermarks(images):        # batch inference
#     images = transform_bhwc_float(images)
#     images = normalize_wat(images)

#     outputs = session_wat.run([], {'input':images})[0]
#     outputs = sigmoid(outputs)
#     return np.where(outputs<=0.5)[0]


context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:7777")
print("started")
while True:
        message = socket.recv(copy=False)
        images = np.frombuffer(message,dtype=np.uint8).reshape(-1,512, 512, 3) #BHWC        
        results_fit = check_fit(images)
        if len(results_fit)==0:
            socket.send(np.array([],dtype=np.int32).tobytes())
            continue

        results_wat = check_watermarks(images[results_fit])
        if len(results_wat)==0:
            socket.send(np.array([],dtype=np.int32).tobytes())
            continue
        
        final_results = results_fit[results_wat]
        socket.send(np.int32(final_results).tobytes())
