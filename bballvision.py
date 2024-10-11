from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np
from sort import *
from collections import deque

# Load Video
video_path = 'input_vids/vid29.mp4'
cap = cv2.VideoCapture(video_path)

model = YOLO("best.pt")

# "made" class doesn't work very well
classnames = ["ball", "made", "person", "rim", "shoot"]

# total_attempts is an array to make sure each shooting stance is only counted once
total_attempts = []
total_made = 0

# Counts the frames
frame = 0

# In the format [x_center, y_center, frame]
rim_position = deque(maxlen=30)
ball_position = deque(maxlen=30)

# Tracking of "Shoot" position
shoot_tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
shoot_detected_frames = []

overlay = None

while True:
    success, img = cap.read()
    if not success:
        break

    if overlay is None:
        overlay = np.zeros_like(img, dtype=np.uint8)

    results = model(img, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # Bounding Box and confidence
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            conf = math.ceil(box.conf[0] * 100) / 100

            # Class name
            cls = int(box.cls[0])
            current_class = classnames[cls]

            cx, cy = x1+w // 2, y1+h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Detecting the "shoot" action
            if current_class == "shoot":
                current_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, current_array))
            
            # Check if ball is detected
            if current_class == "ball" and conf>0.4:
                ball_position.append([cx, cy, frame])

            # Check if rim is detected
            if current_class == "rim" and conf>0.4:
                rim_position.append([cx, cy, frame])

            # Draw bounding boxes for debugging
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{current_class} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    results_shoot_tracker = shoot_tracker.update(detections)

    for tracked in results_shoot_tracker:
        x1, y1, x2, y2, id = tracked.astype(int)
        w, h = x2-x1, y2-y1

        # For debugging
        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10) 

        # Counts max once per shooting stance. Need to test for pump fakes and stuff like that
        if total_attempts.count(id) == 0:
            total_attempts.append(id)

    if frame % 5 == 0:

        # Clear the overlay (reset to transparent)
        overlay = np.zeros_like(img, dtype=np.uint8)
        
        for pos in ball_position:
            cx, cy, pos_frame = pos
            if frame - pos_frame <= 30 and pos_frame % 5 == 0:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Blend the overlay onto the main frame
    blended_img = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    frame += 1

    # Display attempts and made shots count on the image
    cv2.putText(img, f'Attempts: {str(len(total_attempts))}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.putText(img, f'Made Shots: {total_made}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Image", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# point1 = (x1, y1) point2 = (x2, y2)
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)