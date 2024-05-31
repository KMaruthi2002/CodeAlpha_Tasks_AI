import cv2
import numpy as np

# Load YOLOv3 configuration and weights
net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")

# Define the classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the confidence threshold
conf_threshold = 0.5

# Load the video stream
cap = cv2.VideoCapture(0)  # Replace '0' with the path to your video file

while True:
    # Capture the frame from the video stream
    ret, frame = cap.read()

    # Only process the frame if we have successfully captured an image
    if ret:
        # Prepare the frame for processing
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # Perform object detection
        net.setInput(blob)
        layer_outputs = net.forward(["yolo_82", "yolo_94", "yolo_106"])

        # Initialize the list of bounding boxes and confidences
        boxes = []
        confidences = []
        class_ids = []

        # Process the output layer by layer
        for output in layer_outputs:
            for detection in output:
                # Get the scores, class_id, and confidence for the current detection
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Only consider detections with a confidence greater than the confidence threshold
                if confidence > conf_threshold:
                    # Calculate the bounding box coordinates for the current detection
                    box = detection[0:4] * np.array([frame.shape[0], frame.shape[1], frame.shape[0], frame.shape[1]])
                    (center_x, center_y, width, height) = box.astype("int")

                    # Adjust the bounding box coordinates to be in the middle of the pixel coordinates
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # Add the bounding box and confidence to their respective lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Perform non-maximum suppression on the bounding boxes to filter out overlapping detections
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)

        # Draw the bounding boxes and labels for the detections
        for i in indices:
            i = i[0]
            box = boxes[i]
            (x, y, w, h) = box
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame
        cv2.imshow("Object Detection and Tracking", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the video stream and close the output window
cap.release()
cv2.destroyAllWindows()
