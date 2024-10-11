from ultralytics import YOLO
import cv2
import math
import cvzone
import numpy as np
from collections import deque



def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_increasing_distances(point, points_array):
    """
    Check if the distances between the given point and the points in the array
    are strictly increasing.

    Args:
    point (tuple): A tuple (x, y) representing the given point.
    points_array (list): A list of tuples [(x_1, y_1), (x_2, y_2), ...].

    Returns:
    bool: True if the distances are strictly increasing, False otherwise.
    """
    x, y = point

    # Calculate distances between (x, y) and all points in the array
    distances = [distance(point, (x_i, y_i)) for (x_i, y_i) in points_array]

    # Check if distances are strictly increasing
    for i in range(1, len(distances)):
        if distances[i] <= distances[i - 1]:
            return False

    return True



# Load Video
video_path = 'input_vids/vid29.mp4'
cap = cv2.VideoCapture(video_path)

model = YOLO("best.pt")

# "made" class doesn't work very well
classnames = ["ball", "made", "person", "rim", "shoot"]

# total_attempts is an array to make sure each shooting stance is only counted once
total_attempts = 0
total_made = 0

# Counts the frames
frame = 0

# In the format [x_center, y_center, frame]
rim_position = deque(maxlen=30)
ball_position = deque(maxlen=30)
shoot_position = deque(maxlen=30)

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
                shoot_position.append([cx, cy, frame])
            
            # Check if ball is detected
            if current_class == "ball" and conf>0.4:
                ball_position.append([cx, cy, frame])

            # Check if rim is detected
            if current_class == "rim" and conf>0.4:
                rim_position.append([cx, cy, frame])

            # Draw bounding boxes for debugging
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{current_class} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    if shoot_position and shoot_position[-1][2] == frame - 5:
        last_ball_pos = [(cx, cy) for cx, cy, frame in list(ball_position)[-5:]]
        if is_increasing_distances((shoot_position[-1][0], shoot_position[-1][1]), last_ball_pos):
            total_attempts += 1


    if frame % 5 == 0:
        # Clear the overlay (reset to transparent)
        overlay = np.zeros_like(img, dtype=np.uint8)
        
        for pos in ball_position:
            cx, cy, pos_frame = pos
            if pos_frame % 5 == 0:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    # Display attempts and made shots count on the image
    cv2.putText(img, f'Attempts: {str(total_attempts)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.putText(img, f'Made Shots: {total_made}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    # Blend the overlay onto the main frame
    blended_img = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    frame += 1

    cv2.imshow("Image", blended_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
