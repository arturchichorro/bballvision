from ultralytics import YOLO
import cv2
import math
import numpy as np
from collections import deque

def distance(p1, p2):
    """
    Args:
    p1 (x,y) and p2 (x,y)
    """
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_increasing_distances(point, points_array):
    """
    Args:
    point (tuple): (x,y)
    points_array (list): List of tuples [(x_1, y_1), (x_2, y_2), ...].

    Returns:
    bool: True if the distances are strictly increasing, False otherwise.
    """

    # Calculate distances between (x, y) and all points in the array
    distances = [distance(point, (x_i, y_i)) for (x_i, y_i) in points_array]

    # Check if distances are strictly increasing
    for i in range(1, len(distances)):
        if distances[i] <= distances[i - 1]:
            return False
    return True

def is_ball_above_rim(ball, rim):
    """
    Args: 
    ball (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    return ball[1] < rim[1]


def is_ball_below_rim(ball, rim):
    """
    Args: 
    ball (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    return ball[1] > rim[3]

def is_made_shot(above_rim, below_rim, rim):
    """
    Args:
    above_rim (cx, cy, frame)
    below_rim (cx, cy, frame)
    rim (x1, y1, x2, y2, frame)
    """
    x1, y1, x2 = rim[0], rim[1], rim[2]
    cx1, cy1, cx2, cy2 = above_rim[0], above_rim[1], below_rim[0], below_rim[1]

    m = (cy2-cy1)/(cx2-cx1)
    b = cy1 - m*cx1
    x = (y1 - b) / m

    return x1 < x and x < x2

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
ball_position = deque(maxlen=30)
shoot_position = deque(maxlen=30)
# In the format [x1, y1, x2, y2, frame]
rim_position = deque(maxlen=30)

ball_above_rim = None

overlay = None

while True:
    success, img = cap.read()
    if not success:
        break

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

                print(f"Ball position: x1: {x1} x2: {x2} y1: {y1} y2: {y2}")
                print(cx, cy)

            # Check if rim is detected
            if current_class == "rim" and conf>0.4:
                rim_position.append([x1, y1, x2, y2, frame])

                print(f"Rim position: x1: {x1} x2: {x2} y1: {y1} y2: {y2}")
                print(cx, cy)
                cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED) # Pink
                cv2.circle(img, (x1, y2), 5, (0, 255, 0), cv2.FILLED) # Some other 
                cv2.circle(img, (5, 5), 5, (255, 0, 0), cv2.FILLED) # Some other 

            # Draw bounding boxes for debugging
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{current_class} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Checks if distance from shoot position and ball keeps increasing after shot attempt
    # Checks if last time "shoot" was detected was five frames ago
    if shoot_position and shoot_position[-1][2] == frame - 5:
        last_ball_pos = [(cx, cy) for cx, cy, frame in list(ball_position)[-5:]]
        if is_increasing_distances((shoot_position[-1][0], shoot_position[-1][1]), last_ball_pos):
            total_attempts += 1

    # This means that ball was above rim (or between lower and higher rim bound) in last frame and is now below rim
    if ball_above_rim and is_ball_below_rim(ball_position[-1], rim_position[-1]):
        if is_made_shot(ball_above_rim, ball_position[-1], rim_position[-1]):
            total_made += 1
        ball_above_rim = None

    # By doing it through an if statement instead of just assignment, the variable ball_above_rim remains true when
    # lower_rim_bound < ball < higher_rim_bound
    if is_ball_above_rim(ball_position[-1], rim_position[-1]):
        ball_above_rim = ball_position[-1]

    # Display attempts and made shots count on the image
    cv2.putText(img, f'Attempts: {str(total_attempts)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    cv2.putText(img, f'Made Shots: {total_made}', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    # Blend the overlay onto the main frame
    blended_img = cv2.addWeighted(img, 1.0, overlay, 1, 0)

    frame += 1

    # Adds circles on ball position every 5 frames
    if overlay is None:
        overlay = np.zeros_like(img, dtype=np.uint8)

    if frame % 5 == 0:
        # Clear the overlay (reset to transparent)
        overlay = np.zeros_like(img, dtype=np.uint8)
        
        for pos in ball_position:
            cx, cy, pos_frame = pos
            if pos_frame % 5 == 0:
                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Image", blended_img)

    # To watch video frame by frame
    # cv2.waitKey(0)

    # To watch video continuosly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
