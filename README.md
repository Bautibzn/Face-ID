# Face-ID
A YOLO AI trained with a dataset to detect faces and save them.

How it works:

First, we use the Python library ultralyrics to call the yolov11n-face model. Then, we modify its data.yaml to use our own dataset. Next, we train the AI to get the best run, then put it into the code and use it. 
Once done, we will use our model to detect faces and, at the same time, use the insightface library to recognise the face. This allows us to detect multiple faces with the camera.py code and then register the faces we want with registro.py. 
