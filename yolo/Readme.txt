1.Run yolov4_pipeline:
    
             python yolov4_pipeline.py


2. Create a triton inference server :

sudo docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/yolo/model_repository:/models nvcr.io/nvidia/tritonserver:21.08-py3 tritonserver --model-repository=/models --strict-model-config=false

3 run client :
   python image_client_yolo_v4.py -m ensemble_dali_yolo_v4 person_dog.jpg 

