# Create a New Environment 

conda create -n object python=3.7.7
conda activate object


pip install tensorflow==1.15.0 --force-reinstall
pip install opencv-python
pip install keras
pip install imageai --upgrade
pip install os-win
pip install glob2


from imageai.Detection import ObjectDetection
import os
import glob

execution_path = r'C:\Users\--name--\Desktop\Python\Object_Detection'
input_path = r'C:\Users\--name--\Desktop\Python\Object_Detection\Input'
output_path = r'C:\Users\--name--\Desktop\Python\Object_Detection\Output'

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

for i, j in zip(glob.glob(input_path + '\*.jpg'), range(0, len(glob.glob(input_path + '\*.jpg')))):
    detections = detector.detectObjectsFromImage(input_image = i, output_image_path = os.path.join(output_path, str(j) + '.jpg'))
    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"])
