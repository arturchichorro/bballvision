import math
import cv2
import os

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

def write_text_with_background(img, text, location, font_face, font_scale, text_color, background_color, thickness):
    
    (tw, th), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    cv2.rectangle(img, (location[0], location[1] - th - baseline), (location[0] + tw, location[1] + baseline), background_color, -1)
    cv2.putText(img, text, location, font_face, font_scale, text_color, thickness)

def get_available_filename(output_dir, base_name, extension):
    counter = 1
    output_path = os.path.join(output_dir, f"{base_name}.{extension}")
    
    while os.path.exists(output_path):
        output_path = os.path.join(output_dir, f"{base_name}{counter}.{extension}")
        counter += 1
    
    return output_path