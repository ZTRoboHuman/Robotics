from ultralytics import YOLO
import cv2
from imageProcess import generate_random_color

yolomodelfile='yolov8n.pt'
onnxfile="./runs/detect/train/weights/best.onnx"
testImg="bus.jpg"

# visualization parameter setup
vis_lableoffsety = -5

# Load the YOLOv8 model
model = YOLO(yolomodelfile)

# Run batched inference on a list of images
results = model([testImg])  # return a list of Results objects
#print(results)

image = cv2.imread(testImg)

# Iterate over the detection results and draw bounding boxes on the image
for result in results:
    classNames = result.names
    for box,cltensor in zip(result.boxes.xyxy,result.boxes.cls):
        x1, y1, x2, y2 = map(int, box)

        color = generate_random_color()
        thickness = 1
        org = (x1, y1+vis_lableoffsety)
        clname = classNames[cltensor.item()]
        cv2.putText(image, clname , org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.imshow('Yolo_v8_image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
