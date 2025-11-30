import cv2
import numpy as np
import mediapipe as mp
from collections import deque


# Calculate bounding box around both eyes
def get_eye_bbox(all_eye_points, w, h, padding=5):
    all_eye_points = np.array(all_eye_points)
    x_min = max(0, int(all_eye_points[:, 0].min()) - padding - 20)
    x_max = min(w, int(all_eye_points[:, 0].max()) + padding + 20)
    y_min = max(0, int(all_eye_points[:, 1].min()) - padding - 10)
    y_max = min(h, int(all_eye_points[:, 1].max()) + padding + 10)
    return x_min, x_max, y_min, y_max


# Draw eye landmarks and return points and frame
def draw_eye_landmarks(landmarks, indices, w, h, frame):
                all_eye_points = []
                all_eye_idx = []
                for idx in indices:
                    landmark = landmarks[idx]
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    all_eye_points.append((x, y))
                    all_eye_idx.append(idx)
                    cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
                return all_eye_points, all_eye_idx, frame


# Find mean of calibration samples
def calculate_mean(samples):
    return (np.mean(samples), True) if samples else (0, False)



cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5)

# left and right eyelid landmark indices
left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 130]
right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 359]

face_indices = [234, 454, 10, 152, 1, 19, 24, 110, 237, 130, 243, 112, 26, 389, 356, 454]
padding = 5

# Calibration and smoothing for rectangle height (vertical)
calibration_height = None
calibration_samples = deque(maxlen=120)
is_calibrated = False
height_history = deque(maxlen=5)

# Calibration for horizontal (left/right)
calibration_horizontal = None
calibration_h_samples = deque(maxlen=120)
is_h_calibrated = False
horizontal_history = deque(maxlen=5)

combined_gaze = None

# Thresholds
up_threshold = 0
down_threshold = 0
left_threshold = 0
right_threshold = 0

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame, (2560, 1440))
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    landmarks_points = results.multi_face_landmarks

    if landmarks_points:       
            landmarks = landmarks_points[0].landmark
            all_eye_points = []
            all_eye_idx = []
            right_eye_points = []
            left_eye_points = []
            
            # Right eye: landmarks 474-477
            for idx in range(473, 478):
                landmark = landmarks[idx]
                x = landmark.x * w
                y = landmark.y * h
                right_eye_points.append((x, y))
            
            # Left eye: landmarks 469-472
            for idx in range(468, 473):
                landmark = landmarks[idx]
                x = landmark.x * w
                y = landmark.y * h
                left_eye_points.append((x, y))
                
            all_eye_points, all_eye_idx, frame = draw_eye_landmarks(landmarks, left_eye_indices + right_eye_indices, w, h, frame)

            # nose bridge point 10
            nose_bridge = landmarks[10]
            nose_bridge_x = int(nose_bridge.x * w)
            nose_bridge_y = int(nose_bridge.y * h)
            cv2.circle(frame, (nose_bridge_x, nose_bridge_y), 3, (0, 255, 255), -1)


            x_min, x_max, y_min, y_max = get_eye_bbox(all_eye_points, w, h, padding)

            # Crop the eye region from the frame
            eye_frame = frame[y_min:y_max, x_min:x_max].copy()

            # Convert right eye points to eye_frame coordinates
            right_eye_points = np.array(right_eye_points) - np.array([x_min, y_min])
            left_eye_points = np.array(left_eye_points) - np.array([x_min, y_min])

            # Create a rectangle with landmarks 160, 153 for left eye
            le_points_list = []
            le_x1 = int(landmarks[160].x * w)
            le_y1 = int(landmarks[160].y * h)
            le_x2 = int(landmarks[153].x * w)
            le_y2 = int(landmarks[153].y * h)
            le_points_list.append((le_x1, le_y1))
            le_points_list.append((le_x2, le_y2))

            # Create a rectangle with landmarks 380, 387 for right eye
            re_points_list = []
            re_x1 = int(landmarks[380].x * w)
            re_y1 = int(landmarks[380].y * h)
            re_x2 = int(landmarks[387].x * w)
            re_y2 = int(landmarks[387].y * h)
            re_points_list.append((re_x1, re_y1))
            re_points_list.append((re_x2, re_y2))

            # Calculate rectangle heights (vertical detection)
            left_rect_height = abs(le_y2 - le_y1)
            right_rect_height = abs(re_y2 - re_y1)
            avg_rect_height = (left_rect_height + right_rect_height) / 2

            # Calculate iris centers in original frame coordinates
            left_iris_center_x = np.mean([landmarks[idx].x for idx in range(468, 473)]) * w
            left_iris_center_y = np.mean([landmarks[idx].y for idx in range(468, 473)]) * h
            right_iris_center_x = np.mean([landmarks[idx].x for idx in range(473, 478)]) * w
            right_iris_center_y = np.mean([landmarks[idx].y for idx in range(473, 478)]) * h

            # Calculate horizontal position relative to eye rectangle
            # For right eye: distance from iris center to left edge (re_x1)
            right_eye_rect_width = abs(re_x2 - re_x1)
            right_iris_offset = (right_iris_center_x - re_x1) / right_eye_rect_width if right_eye_rect_width > 0 else 0.5
            
            # For left eye: distance from iris center to left edge (le_x1)
            left_eye_rect_width = abs(le_x2 - le_x1)
            left_iris_offset = (left_iris_center_x - le_x1) / left_eye_rect_width if left_eye_rect_width > 0 else 0.5
            
            # Average both eyes for horizontal position (0 = left, 0.5 = center, 1 = right)
            avg_horizontal_position = (left_iris_offset + right_iris_offset) / 2

            # Smooth measurements
            height_history.append(avg_rect_height)
            smoothed_height = np.mean(height_history)
            
            horizontal_history.append(avg_horizontal_position)
            smoothed_horizontal = np.mean(horizontal_history)

            # Calibration mode
            if not is_calibrated:
                calibration_samples.append(smoothed_height)
                
                # Display calibration instructions
                cv2.putText(frame, "CALIBRATION: Look straight ahead", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.putText(frame, f"Samples: {len(calibration_samples)}/30", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_samples) == 60:
                    calibration_height, is_calibrated = calculate_mean(calibration_samples)
                    up_threshold = calibration_height * 1.05
                    down_threshold = calibration_height * 0.95
                    print(f"Vertical Calibrated! Center height: {calibration_height:.2f}")
                    print(f"Up threshold: {up_threshold:.2f}, Down threshold: {down_threshold:.2f}")
            
            elif not is_h_calibrated:
                calibration_h_samples.append(smoothed_horizontal)
                
                # Display horizontal calibration
                cv2.putText(frame, "CALIBRATION: Keep looking straight", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                cv2.putText(frame, f"H-Samples: {len(calibration_h_samples)}/30", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_h_samples) == 60:
                    calibration_horizontal, is_h_calibrated = calculate_mean(calibration_h_samples)
                    # Threshold: ±15% from center position
                    left_threshold = calibration_horizontal - 0.15
                    right_threshold = calibration_horizontal + 0.15
                    print(f"Horizontal Calibrated! Center: {calibration_horizontal:.3f}")
                    print(f"Left threshold: {left_threshold:.3f}, Right threshold: {right_threshold:.3f}")
            
            else:
                # Determine vertical gaze (up/down)
                if smoothed_height > up_threshold:
                    vertical_gaze = "UP"
                    v_color = (0, 255, 0)
                elif smoothed_height < down_threshold:
                    vertical_gaze = "DOWN"
                    v_color = (0, 0, 255)
                else:
                    vertical_gaze = "CENTER"
                    v_color = (255, 255, 0)

                # Determine horizontal gaze (left/right)
                if smoothed_horizontal < left_threshold:
                    horizontal_gaze = "LEFT"
                    h_color = (255, 0, 255)  # Magenta
                elif smoothed_horizontal > right_threshold:
                    horizontal_gaze = "RIGHT"
                    h_color = (0, 165, 255)  # Orange
                else:
                    horizontal_gaze = "CENTER"
                    h_color = (255, 255, 0)

                # Combined gaze direction
                if vertical_gaze == "CENTER" and horizontal_gaze == "CENTER":
                    combined_gaze = "CENTER"
                    combined_color = (255, 255, 255)
                else:
                    combined_gaze = f"{vertical_gaze} {horizontal_gaze}"
                    combined_color = (0, 255, 255)

                # Display gaze directions
                cv2.putText(frame, combined_gaze, 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, combined_color, 4)
                
                cv2.putText(frame, f"V: {vertical_gaze}", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, v_color, 2)
                cv2.putText(frame, f"H: {horizontal_gaze}", 
                           (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, h_color, 2)
                
                # Display metrics
                cv2.putText(frame, f"Height: {smoothed_height:.1f} / {calibration_height:.1f}", 
                           (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame, f"H-Pos: {smoothed_horizontal:.3f} / {calibration_horizontal:.3f}", 
                           (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Visual indicator bars
                # Vertical bar
                v_bar_x = 50
                v_bar_y = 350
                bar_width = 400
                bar_height = 20
                
                cv2.rectangle(frame, (v_bar_x, v_bar_y), (v_bar_x + bar_width, v_bar_y + bar_height), 
                             (100, 100, 100), 2)
                v_fill = int((smoothed_height / calibration_height) * bar_width)
                v_fill = max(0, min(bar_width, v_fill))
                cv2.rectangle(frame, (v_bar_x, v_bar_y), (v_bar_x + v_fill, v_bar_y + bar_height), 
                             v_color, -1)
                
                # Horizontal bar
                h_bar_y = 380
                cv2.rectangle(frame, (v_bar_x, h_bar_y), (v_bar_x + bar_width, h_bar_y + bar_height), 
                             (100, 100, 100), 2)
                h_fill = int(smoothed_horizontal * bar_width)
                h_fill = max(0, min(bar_width, h_fill))
                cv2.rectangle(frame, (v_bar_x, h_bar_y), (v_bar_x + h_fill, h_bar_y + bar_height), 
                             h_color, -1)
                
                # Mark center on horizontal bar
                center_mark = int(calibration_horizontal * bar_width)
                cv2.line(frame, (v_bar_x + center_mark, h_bar_y), 
                        (v_bar_x + center_mark, h_bar_y + bar_height), (255, 255, 255), 3)

            # Mid point for rectangle cross
            r_mid_x = (re_x1 + re_x2) // 2
            r_mid_y = (re_y1 + re_y2) // 2
            l_mid_x = (le_x1 + le_x2) // 2
            l_mid_y = (le_y1 + le_y2) // 2

            # Convert to eye_frame coordinates
            r_mid_x -= x_min
            r_mid_y -= y_min
            l_mid_x -= x_min
            l_mid_y -= y_min
            ef_re_y2 = re_y2 - y_min
            ef_re_y1 = re_y1 - y_min
            ef_re_x1 = re_x1 - x_min
            ef_re_x2 = re_x2 - x_min
            ef_le_y2 = le_y2 - y_min
            ef_le_y1 = le_y1 - y_min
            ef_le_x1 = le_x1 - x_min
            ef_le_x2 = le_x2 - x_min

            # Cross in the middle of rectangles
            cv2.line(eye_frame, (r_mid_x, ef_re_y1), (r_mid_x, ef_re_y2), (255, 0, 0), 1)
            cv2.line(eye_frame, (ef_re_x1, r_mid_y), (ef_re_x2, r_mid_y), (255, 0, 0), 1)
            cv2.line(eye_frame, (l_mid_x, ef_le_y1), (l_mid_x, ef_le_y2), (255, 0, 0), 1)
            cv2.line(eye_frame, (ef_le_x1, l_mid_y), (ef_le_x2, l_mid_y), (255, 0, 0), 1)

            # Draw rectangles
            le_points = np.array(le_points_list) - np.array([x_min, y_min])
            re_points = np.array(re_points_list) - np.array([x_min, y_min])
            
            cv2.rectangle(eye_frame, tuple(le_points[0]), tuple(le_points[1]), (255, 0, 0), 1)
            cv2.rectangle(eye_frame, tuple(re_points[0]), tuple(re_points[1]), (255, 0, 0), 1)

            # Display measurements
            cv2.putText(eye_frame, f"L_H: {left_rect_height:.0f}", 
                       tuple(le_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(eye_frame, f"L_Pos: {left_iris_offset:.2f}", 
                       (tuple(le_points[0])[0], tuple(le_points[0])[1] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            cv2.putText(eye_frame, f"R_H: {right_rect_height:.0f}", 
                       tuple(re_points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(eye_frame, f"R_Pos: {right_iris_offset:.2f}", 
                       (tuple(re_points[0])[0], tuple(re_points[0])[1] + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Draw all eye landmarks
            all_eye_points = np.array(all_eye_points) - np.array([x_min, y_min])
            for point in all_eye_points:
                cv2.circle(eye_frame, tuple(point), 1, (0, 255, 255), -1)
                idx = all_eye_idx[all_eye_points.tolist().index(point.tolist())]
                if idx in (160, 153, 380, 387):
                    cv2.putText(eye_frame, str(idx), tuple(point), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Draw iris centers
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(right_eye_center), 3, (0, 0, 255), -1)

            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(left_eye_center), 3, (0, 0, 255), -1)
                
            eye_frame = cv2.resize(eye_frame, (eye_frame.shape[1]+50, eye_frame.shape[0]+50))          
            cv2.imshow('Cropped Eyes', eye_frame)

    screen_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw four quadrants using rectangles
    if combined_gaze == "UP LEFT":
        cv2.rectangle(screen_frame, (0, 0), (640, 360), (0, 255, 0), -1)          # Top-Left
    elif combined_gaze == "UP RIGHT":
        cv2.rectangle(screen_frame, (640, 0), (1280, 360), (0, 255, 0), -1)      # Top-Right
    elif combined_gaze == "DOWN LEFT":
        cv2.rectangle(screen_frame, (0, 360), (640, 720), (0, 0, 255), -1)    # Bottom-Left
    elif combined_gaze == "DOWN RIGHT":
        cv2.rectangle(screen_frame, (640, 360), (1280, 720), (0, 0, 255), -1)  # Bottom-Right
    else:
        cv2.rectangle(screen_frame, (0, 0), (640, 360), (50, 50, 50), -1)          # Top-Left
        cv2.rectangle(screen_frame, (640, 0), (1280, 360), (80, 80, 80), -1)      # Top-Right
        cv2.rectangle(screen_frame, (0, 360), (640, 720), (110, 110, 110), -1)    # Bottom-Left
        cv2.rectangle(screen_frame, (640, 360), (1280, 720), (140, 140, 140), -1)  # Bottom-Right
    
    
    cv2.imshow('Screen Quadrants', screen_frame)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Eye Tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and is_calibrated and is_h_calibrated:
        # Recalibrate both
        is_calibrated = False
        is_h_calibrated = False
        calibration_samples.clear()
        calibration_h_samples.clear()
        height_history.clear()
        horizontal_history.clear()
        print("Recalibrating both axes...")

cam.release()
cv2.destroyAllWindows()