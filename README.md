# anti_sus  
Outlier detection in embeddings  
download clip_visual.onnx from [here](https://github.com/qwertyforce/anti_sus/releases/tag/clip_onnx_model)  
download model_wat.onnx from [here](https://github.com/qwertyforce/anti_sus/releases/tag/wat_model_0.1)

[how it works](https://github.com/qwertyforce/anti_sus/blob/main/automatic_image_mining.md)  
  
get features with [image_text_features_web](https://github.com/qwertyforce/image_text_features_web) copy them to folder ./clean/  

train_gmm.py -> trains gmm with features from ./clean  
testing.ipynb -> notebook for comparing distributions of ./clean/ and ./test/ for manual adjusting of threshold  
anti_sus.py -> zeromq server for filtering outlier images. Receives batch of rgb numpy images, returns indexes of good images.  
It has 2 step filtering:
- gmm score threshold  
- watermark detection (filters images with watermarks, trained on  [scenery_watermarks](https://github.com/qwertyforce/scenery_watermarks))

### Docker
```docker build -t qwertyforce/anti_sus_nomad:1.0.0 --network host -t qwertyforce/anti_sus_nomad:latest ./```  
```docker run -d --network host --name anit_sus_nomad qwertyforce/anti_sus_nomad:1.0.0```  
