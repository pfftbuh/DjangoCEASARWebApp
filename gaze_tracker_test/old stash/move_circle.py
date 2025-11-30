import cv2
import numpy as np
import keyboard  

def move_circle(frame, position, radius=20, color=(0, 255, 0)):
    cv2.circle(frame, position, radius, color, -1)
    return frame

circle_position = [320, 240]
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    
    # Check multiple keys simultaneously
    move_speed = 5
    
    if keyboard.is_pressed('w'):  # Move up
        circle_position[1] = max(0, circle_position[1] - move_speed)
    if keyboard.is_pressed('s'):  # Move down
        circle_position[1] = min(height, circle_position[1] + move_speed)
    if keyboard.is_pressed('a'):  # Move left
        circle_position[0] = max(0, circle_position[0] - move_speed)
    if keyboard.is_pressed('d'):  # Move right
        circle_position[0] = min(width, circle_position[0] + move_speed)
    
    if keyboard.is_pressed('q'):
        break

    frame_with_circle = move_circle(frame, tuple(circle_position))
    cv2.imshow('Circle Tracker', frame_with_circle)
    
    cv2.waitKey(10)  # Still needed for window refresh

cap.release()
cv2.destroyAllWindows()