# Workshop 2: Object Detection Using Webcam with YOLOv4

## Aim

To write a Python program using OpenCV and YOLOv4 to perform real time object detection through a laptop webcam and identify objects present in the captured video frames.

## Software Required

* Anaconda Python 3.7 or above
* OpenCV (`opencv-python`)
* NumPy
* Matplotlib
* Jupyter Notebook or any Python IDE
* YOLOv4 Configuration File (`yolov4.cfg`)
* YOLOv4 Weights File (`yolov4.weights`)
* COCO Class Names File (`coco.names`)

## Algorithm

### I) Import Required Libraries

* Step 1: Import OpenCV for image and video processing.
* Step 2: Import NumPy for numerical computations.
* Step 3: Import Matplotlib for displaying images.

### II) Load YOLOv4 Model

* Step 1: Load the YOLOv4 configuration file using `cv2.dnn.readNet()`.
* Step 2: Load the pretrained YOLOv4 weights.
* Step 3: Load COCO dataset class labels from `coco.names`.
* Step 4: Retrieve output layers of the YOLO network.

### III) Define Object Detection Function

* Step 1: Convert the input image into a blob using `cv2.dnn.blobFromImage()`.
* Step 2: Feed the blob to the YOLO network.
* Step 3: Perform forward propagation to obtain detections.
* Step 4: Extract confidence scores and class IDs.
* Step 5: Apply Non Maximum Suppression (NMS) to remove overlapping boxes.
* Step 6: Draw bounding boxes and class labels around detected objects.
* Step 7: Return the processed image.

### IV) Capture Image from Webcam

* Step 1: Initialize webcam using `cv2.VideoCapture(0)`.
* Step 2: Capture a frame from the webcam.
* Step 3: Store the captured frame in a variable.
* Step 4: Release the webcam resource.

### V) Perform Object Detection

* Step 1: Pass the captured image to the object detection function.
* Step 2: Detect objects present in the image.
* Step 3: Draw bounding boxes with labels and confidence scores.
* Step 4: Store the detection result in a separate variable.

### VI) Display Detection Results

* Step 1: Convert the image from BGR to RGB format.
* Step 2: Display the processed image using Matplotlib.
* Step 3: Show detected objects with their labels.

## PROGRAM

### SANJAI L

### 212223230184

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv4 files
weights_path = "yolov4.weights"
config_path = "yolov4.cfg"
names_path = "coco.names"

net = cv2.dnn.readNet(weights_path, config_path)

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Object Detection Function
def detect_objects(frame):

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1/255.0,
        size=(416,416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:

        for i in indexes.flatten():

            x,y,w,h = boxes[i]

            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return frame

# Capture image from webcam
cap = cv2.VideoCapture(0)

ret, captured_image = cap.read()

cap.release()

# Perform detection
detected_image = detect_objects(captured_image)

# Display result
plt.figure(figsize=(12,8))
plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB))
plt.title("YOLOv4 Object Detection")
plt.axis("off")
plt.show()
```

## OUTPUT

<img width="766" height="606" alt="image" src="https://github.com/user-attachments/assets/407438cb-987f-4b29-a3e6-5adc512c2f56" />

<img width="875" height="687" alt="image" src="https://github.com/user-attachments/assets/09f49ae9-abde-4c96-8ff6-d7933da3acb5" />



## RESULT

Thus, object detection using a webcam was successfully implemented using the YOLOv4 model and OpenCV. The detected objects were identified and displayed with bounding boxes, labels, and confidence scores.
