            # # Attempt: when ball is moving towards the rim from above
            # if prev_ball_y < rim_top_y and current_ball_y > rim_top_y:
            #     totalAttempts += 1
            #     print(f"Shot Attempt: {totalAttempts} attempts")

            # Made shot: when ball crosses rim's bottom boundary


from ultralytics import YOLO
import cv2
import math
import numpy as np
from sort import *

# Load Video
video_path = 'input_vids/vid29.mp4'
cap = cv2.VideoCapture(video_path)

model = YOLO("best.pt")  # Your trained model

# "made" class doesn't work very well
classNames = ["ball", "made", "person", "rim", "shoot"]

totalAttempts = 0
totalMadeShots = 0

# Tracking of "Shoot" position
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLO model to detect objects (ball, rim, person)
    results = model(img, stream=True)

    rim_detected = False
    ball_detected = False

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            
            # BOUNDING BOX
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil(box.conf[0] * 100) / 100

            # CLASS NAME
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Check if rim is detected
            if currentClass == "rim":
                rim_detected = True
            
            # Check if ball is detected
            if currentClass == "ball":
                ball_detected = True

            # Detecting the "shoot" action
            if currentClass == "shoot":
                totalAttempts += 1

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

            # Draw bounding boxes for debugging
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{currentClass} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    resultsTracker = tracker.update(detections)

    

    # Display attempts and made shots count on the image
    cv2.putText(img, f'Attempts: {totalAttempts}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.putText(img, f'Made Shots: {totalMadeShots}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
