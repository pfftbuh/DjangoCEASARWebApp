import math
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import json
import os


SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Calculate bounding box around both eyesq
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

def calculate_pts_mean(pts):
    return np.mean(pts, axis=0).astype(int)

# Calculate iris centers in original frame coordinates
def get_iris_center(landmarks, indices, w, h):
    x = np.mean([landmarks[idx].x for idx in indices]) * w
    y = np.mean([landmarks[idx].y for idx in indices]) * h
    return x, y

def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])

# JSON Calibration Functions
def save_calibration(filename='calibration_data.json'):
    """Save calibration data to JSON file"""
    calibration_data = {
        'calibration_height_up': float(calibration_height_up),
        'calibration_height_center': float(calibration_height_center),
        'calibration_height_down': float(calibration_height_down),
        'calibration_horizontal_left': float(calibration_horizontal_left),
        'calibration_horizontal_center': float(calibration_horizontal_center),
        'calibration_horizontal_right': float(calibration_horizontal_right),
        'calibration_offset_yaw': float(calibration_offset_yaw),
        'calibration_offset_pitch': float(calibration_offset_pitch),
        'up_threshold': float(up_threshold),
        'down_threshold': float(down_threshold),
        'left_threshold': float(left_threshold),
        'right_threshold': float(right_threshold),
        'distance_threshold': float(distance_threshold),
        'is_up_calibrated': is_up_calibrated,
        'is_center_calibrated': is_center_calibrated,
        'is_down_calibrated': is_down_calibrated,
        'is_left_calibrated': is_left_calibrated,
        'is_h_center_calibrated': is_h_center_calibrated,
        'is_right_calibrated': is_right_calibrated
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=4)
        print(f"Calibration data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return False

def load_calibration(filename='calibration_data.json'):
    """Load calibration data from JSON file"""
    if not os.path.exists(filename):
        print(f"No calibration file found at {filename}")
        return False
    
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        
        global calibration_height_up, calibration_height_center, calibration_height_down
        global calibration_horizontal_left, calibration_horizontal_center, calibration_horizontal_right
        global calibration_offset_yaw, calibration_offset_pitch
        global up_threshold, down_threshold, left_threshold, right_threshold, distance_threshold
        global is_up_calibrated, is_center_calibrated, is_down_calibrated
        global is_left_calibrated, is_h_center_calibrated, is_right_calibrated
        global calibration_stage
        
        calibration_height_up = data['calibration_height_up']
        calibration_height_center = data['calibration_height_center']
        calibration_height_down = data['calibration_height_down']
        calibration_horizontal_left = data['calibration_horizontal_left']
        calibration_horizontal_center = data['calibration_horizontal_center']
        calibration_horizontal_right = data['calibration_horizontal_right']
        calibration_offset_yaw = data['calibration_offset_yaw']
        calibration_offset_pitch = data['calibration_offset_pitch']
        up_threshold = data['up_threshold']
        down_threshold = data['down_threshold']
        left_threshold = data['left_threshold']
        right_threshold = data['right_threshold']
        distance_threshold = data['distance_threshold']
        is_up_calibrated = data['is_up_calibrated']
        is_center_calibrated = data['is_center_calibrated']
        is_down_calibrated = data['is_down_calibrated']
        is_left_calibrated = data['is_left_calibrated']
        is_h_center_calibrated = data['is_h_center_calibrated']
        is_right_calibrated = data['is_right_calibrated']
        
        # Set calibration stage to tracking mode
        calibration_stage = 6
        
        print(f"Calibration data loaded from {filename}")
        print(f"Vertical: Up={calibration_height_up:.2f}, Center={calibration_height_center:.2f}, Down={calibration_height_down:.2f}")
        print(f"Horizontal: Left={calibration_horizontal_left:.3f}, Center={calibration_horizontal_center:.3f}, Right={calibration_horizontal_right:.3f}")
        return True
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False

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

# y width in centimeters
y_dist = [240, 132, 350, 560, 200]
cm_dist = [50, 90, 38, 20, 70]
dist_coff = np.polyfit(y_dist, cm_dist, deg=2)

# Face Axis Key Indexes
KEY_FACE_LANDMARKS = {
    "left":234,
    "right": 454,
    "top": 10,
    "bottom":152,
    "front":1
}

# ========================== STATE & CALIBRATION VARIABLES ==========================


# Ray smoothing for head pose
ray_origins = deque(maxlen=5)
ray_directions = deque(maxlen=5)
ray_length = 50

# Gaze Circle Position
iris_gaze_position = (640, 360)  # Initial position at the center of a 1280x720 frame

# Utility deques for smoothing gaze positions and colors
face_gaze_position_history = deque(maxlen=5)
iris_gaze_position_history = deque(maxlen=5)
combined_color_history = deque(maxlen=5)

# Calibration samples deques
calibration_distance_samples = deque(maxlen=60)
calibration_left_samples = deque(maxlen=60)
calibration_h_center_samples = deque(maxlen=60)
calibration_right_samples = deque(maxlen=60)
calibration_up_samples = deque(maxlen=60)
calibration_center_samples = deque(maxlen=60)
calibration_down_samples = deque(maxlen=60)

# Calibration and smoothing for rectangle height (vertical)
calibration_height_up = 0.0
calibration_height_center = 0.0
calibration_height_down = 0.0

is_up_calibrated = False
is_center_calibrated = False
is_down_calibrated = False
height_history = deque(maxlen=5)

# Calibration for horizontal (left/right)
calibration_horizontal_left = 0.0
calibration_horizontal_center = 0.0
calibration_horizontal_right = 0.0

calibration_offset_yaw = 0
calibration_offset_pitch = 0

is_left_calibrated = False
is_h_center_calibrated = False
is_right_calibrated = False
horizontal_history = deque(maxlen=5)

# Combined gaze direction string
combined_gaze = None

# Thresholds (will be calculated from calibration data)
up_threshold = 0
down_threshold = 0
left_threshold = 0
right_threshold = 0
distance_threshold = 0

calibration_stage = -1  # -1=waiting to start, 0=up, 1=center, 2=down, 3=left, 4=h_center, 5=right, 6=done

# Try to load existing calibration on startup
print("Checking for existing calibration...")
if load_calibration():
    print("Press 'r' to recalibrate or continue with loaded calibration")
else:
    print("No calibration found. Press 'c' to start calibration.")

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
            right_eye_points, right_eye_idx, frame = draw_eye_landmarks(landmarks, range(473, 478), w, h, frame)
            
            # Left eye: landmarks 469-472
            left_eye_points, left_eye_idx, frame = draw_eye_landmarks(landmarks, range(468, 473), w, h, frame)
                
            # Draw all eye landmarks
            all_eye_points, all_eye_idx, frame = draw_eye_landmarks(landmarks, left_eye_indices + right_eye_indices, w, h, frame)


            # ============================================= FACE AXIS ALGORITHMS =============================================== #

            # region FACE AXIS ALGORITHMS

            def landmark_to_np(landmark, w, h):
                return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

            key_points = {}
            for name, idx in KEY_FACE_LANDMARKS.items():
                pt = landmark_to_np(landmarks[idx], w, h)
                key_points[name] = pt
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x,y), 10, (0,0,255), -1)
            
            left_pt = key_points['left']
            right_pt = key_points['right']
            bottom_pt = key_points['bottom']
            top_pt = key_points['top']
            front_pt = key_points['front']

            # Head Distance Estimation (simple approximation)
            pointLeft = landmark_to_np(landmarks[145], w, h)
            pointRight = landmark_to_np(landmarks[374], w, h)
            width_pts = math.sqrt((pointLeft[0] - pointRight[0])**2 + (pointLeft[1] - pointRight[1])**2)
            A, B, C = dist_coff
            distanceCM = A * width_pts**2 + B * width_pts + C

            # Write width_pts in frame
            cv2.putText(frame, f"Distance CM: {distanceCM:.2f}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

            # Oriented axes based on head geometry
            right_axis = (right_pt - left_pt)
            right_axis /= np.linalg.norm(right_axis)

            up_axis = (top_pt - bottom_pt)
            up_axis /= np.linalg.norm(up_axis)

            forward_axis = np.cross(right_axis, up_axis)
            forward_axis /= np.linalg.norm(forward_axis)

            # Flip to ensure forward vector comes out of the face
            forward_axis = -forward_axis

            # Compute center of the head
            center = (left_pt + right_pt + top_pt + bottom_pt + front_pt) / 5

            # Half-sizes (width, height, depth)
            half_width = np.linalg.norm(right_pt - left_pt) / 2
            half_height = np.linalg.norm(top_pt - bottom_pt) / 2
            half_depth = 80

            ray_origins.append(center)
            ray_directions.append(forward_axis)

            avg_origin = np.mean(ray_origins, axis=0)
            avg_direction = np.mean(ray_directions, axis=0)
            avg_direction /= np.linalg.norm(avg_direction)

            # Reference forward direction (camera looking straight ahead)
            reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

            # Draw smoothed ray
            ray_length = 1.5 * half_depth
            ray_end = avg_origin - avg_direction * ray_length
            
            ray_padding_y = 25
            avg_origin_x, avg_origin_y = project(avg_origin)
            ray_end_x, ray_end_y = project(ray_end)

            # Add the mean of all eye points y to y of avg_origin and ray_end 
            mean_eye_point = np.mean(all_eye_points, axis=0)
            avg_origin_y = int((mean_eye_point[1] + avg_origin_y) / 2)
            ray_end_y = int((mean_eye_point[1] + ray_end_y) / 2)

            # Draw ray in frame
            cv2.line(frame, (avg_origin_x, avg_origin_y - ray_padding_y), (ray_end_x , ray_end_y - ray_padding_y), (0, 0, 255), 2)

            # ESTIMATE GAZE POSITION
            # Horizontal (yaw) angle from reference (project onto XZ plane)
            xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
            xz_proj /= np.linalg.norm(xz_proj)
            yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
            if avg_direction[0] < 0:
                yaw_rad = -yaw_rad  # left is negative

            # Vertical (pitch) angle from reference (project onto YZ plane)
            yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
            yz_proj /= np.linalg.norm(yz_proj)
            pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
            if avg_direction[1] > 0:
                pitch_rad = -pitch_rad  # up is positive

            # Convert to degrees and re-center around 0
            yaw_deg = np.degrees(yaw_rad)
            pitch_deg = np.degrees(pitch_rad)

            #this results in the center being 180, +10 left = -170, +10 right = +170

            #convert left rotations to 0-180
            if yaw_deg < 0:
                yaw_deg = abs(yaw_deg)
            elif yaw_deg < 180:
                yaw_deg = 360 - yaw_deg

            if pitch_deg < 0:
                pitch_deg = 360 + pitch_deg

            raw_yaw_deg = yaw_deg
            raw_pitch_deg = pitch_deg

            #specify degrees at which screen border will be reached
            yawDegrees = 20 # x degrees left or right
            pitchDegrees = 10 # x degrees up or down
            
            # leftmost pixel position must correspond to 180 - yaw degrees
            # rightmost pixel position must correspond to 180 + yaw degrees
            # topmost pixel position must correspond to 180 + pitch degrees
            # bottommost pixel position must correspond to 180 - pitch degrees

            # Apply calibration offsets
            yaw_deg += calibration_offset_yaw
            pitch_deg += calibration_offset_pitch

            screen_frame_w, screen_frame_h = SCREEN_WIDTH, SCREEN_HEIGHT
            
            # Map to full screen resolution
            screen_x = int((screen_frame_w - ((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * screen_frame_w))
            screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * screen_frame_h)
            
            
            # Write gaze position on bottom right frame
            cv2.putText(frame, f"Gaze Pos: ({screen_x}, {screen_y})", (w - 300, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            face_gaze_position = (screen_x, screen_y)
            face_gaze_position_history.append(face_gaze_position)
            
            if len(face_gaze_position_history) > 0:
                # Smooth gaze position
                smoothed_x = int(np.mean([pos[0] for pos in face_gaze_position_history] + [face_gaze_position[0]]))
                smoothed_y = int(np.mean([pos[1] for pos in face_gaze_position_history] + [face_gaze_position[1]]))
                face_gaze_position = (smoothed_x, smoothed_y)
            
            # endregion FACE AXIS ALGORITHMS

            # ============================================= FACE AXIS ALGORITHMS =============================================== #


            # ============================================= EYE FRAME DRAWING =============================================== #

            # Calculate bounding box around both eyes
            all_eye_points = np.array(all_eye_points)
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

            # Calculate iris centers
            left_iris_center_x, left_iris_center_y = get_iris_center(landmarks, range(468, 473), w, h)
            right_iris_center_x, right_iris_center_y = get_iris_center(landmarks, range(473, 478), w, h)

            # Calculate horizontal position relative to eye rectangle
            right_eye_rect_width = abs(re_x2 - re_x1)
            right_iris_offset = (right_iris_center_x - re_x1) / right_eye_rect_width if right_eye_rect_width > 0 else 0.5
            
            left_eye_rect_width = abs(le_x2 - le_x1)
            left_iris_offset = (left_iris_center_x - le_x1) / left_eye_rect_width if left_eye_rect_width > 0 else 0.5
            
            avg_horizontal_position = (left_iris_offset + right_iris_offset) / 2

            # Smooth measurements
            height_history.append(avg_rect_height)
            smoothed_height = np.mean(height_history)
            
            horizontal_history.append(avg_horizontal_position)
            smoothed_horizontal = np.mean(horizontal_history)

            # Convert to eye_frame coordinates
            ef_ray_end = ray_end - np.array([x_min, y_min, 0])
            ef_avg_origin = avg_origin - np.array([x_min, y_min, 0])

            # Draw ray in eye_frame
            cv2.line(eye_frame, project(ef_avg_origin), project(ef_ray_end), (15, 255, 0), 3)

            # Mid point for rectangle cross
            r_mid_x = (re_x1 + re_x2) // 2
            r_mid_y = (re_y1 + re_y2) // 2
            l_mid_x = (le_x1 + le_x2) // 2
            l_mid_y = (le_y1 + le_y2) // 2

            # Convert to eye_frame y coordinates
            def pos_converter(pos, pos_min):
                return pos - pos_min
            
            r_mid_x = pos_converter(r_mid_x, x_min)
            r_mid_y = pos_converter(r_mid_y, y_min)
            l_mid_x = pos_converter(l_mid_x, x_min)
            l_mid_y = pos_converter(l_mid_y, y_min)
            ef_re_y2 = pos_converter(re_y2, y_min)
            ef_re_y1 = pos_converter(re_y1, y_min)
            ef_re_x1 = pos_converter(re_x1, x_min)
            ef_re_x2 = pos_converter(re_x2, x_min)
            ef_le_y2 = pos_converter(le_y2, y_min)
            ef_le_y1 = pos_converter(le_y1, y_min)
            ef_le_x1 = pos_converter(le_x1, x_min)
            ef_le_x2 = pos_converter(le_x2, x_min)


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
            # Place left eye info at the top left of the eye frame
            cv2.putText(eye_frame, f"L_H: {left_rect_height:.0f}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(eye_frame, f"L_Pos: {left_iris_offset:.2f}", 
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Place right eye info on the top right of the eye frame
            right_text_x = eye_frame.shape[1] - 170  # adjust as needed for padding
            right_text_y = 30
            cv2.putText(eye_frame, f"R_H: {right_rect_height:.0f}", 
                        (right_text_x, right_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(eye_frame, f"R_Pos: {right_iris_offset:.2f}", 
                        (right_text_x, right_text_y + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

            # Draw iris centers
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(right_eye_center), 3, (0, 0, 255), -1)

            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(left_eye_center), 3, (0, 0, 255), -1)
            
            # Fixed Dimensions
            eye_frame = cv2.resize(eye_frame, (900, 400))         
            cv2.imshow('Cropped Eyes', eye_frame)

            # ============================================= EYE FRAME DRAWING =============================================== #


            # ============================================ CALIBRATION & GAZE DETECTION =============================================== #

            #region Calibration & Gaze Detection

            # Multi-stage calibration
            if calibration_stage == -1:
                cv2.putText(frame, "Press 'c' to start CALIBRATION for UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    calibration_offset_yaw = 180 - raw_yaw_deg
                    calibration_offset_pitch = 180 - raw_pitch_deg
                    calibration_stage = 0
                    print("=== CALIBRATION STARTED ===")
                
                
            if calibration_stage == 0:  # Calibrate UP
                if len(calibration_up_samples) < 60:
                    calibration_up_samples.append(smoothed_height)
                if len(calibration_distance_samples) < 10:
                    calibration_distance_samples.append(distanceCM)                
                cv2.putText(frame, "CALIBRATION: Look UP", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                cv2.putText(frame, f"Samples: {len(calibration_up_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_up_samples) == 60:
                    calibration_height_up = np.mean(calibration_up_samples)
                    print(f"UP Calibrated! Height: {calibration_height_up:.2f}")
                    cv2.putText(frame, "Press 'c' to continue to CENTER calibration", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        is_up_calibrated = True
                        calibration_stage = 1
            
            elif calibration_stage == 1:  # Calibrate CENTER (vertical)
                if len(calibration_center_samples) < 60:
                    calibration_center_samples.append(smoothed_height)
                if len(calibration_distance_samples) < 20:
                    calibration_distance_samples.append(distanceCM)  
                cv2.putText(frame, "CALIBRATION: Look CENTER", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                cv2.putText(frame, f"Samples: {len(calibration_center_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_center_samples) == 60:
                    calibration_height_center, is_center_calibrated = calculate_mean(calibration_center_samples)
                    print(f"CENTER Calibrated! Height: {calibration_height_center:.2f}")
                    cv2.putText(frame, "Press 'c' to continue to DOWN calibration", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        calibration_stage = 2
            
            elif calibration_stage == 2:  # Calibrate DOWN
                if len(calibration_down_samples) < 60:
                    calibration_down_samples.append(smoothed_height)
                if len(calibration_distance_samples) < 30:
                    calibration_distance_samples.append(distanceCM)  
                cv2.putText(frame, "CALIBRATION: Look DOWN", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Samples: {len(calibration_down_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_down_samples) == 60:
                    calibration_height_down, is_down_calibrated = calculate_mean(calibration_down_samples)
                    # Calculate thresholds as midpoints
                    up_threshold = (calibration_height_up + calibration_height_center) / 2
                    down_threshold = (calibration_height_center + calibration_height_down) / 2
                    print(f"DOWN Calibrated! Height: {calibration_height_down:.2f}")
                    print(f"Vertical thresholds - Up: {up_threshold:.2f}, Down: {down_threshold:.2f}")
                    cv2.putText(frame, "Press 'c' to continue to LEFT calibration", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        calibration_stage = 3
                        
            
            elif calibration_stage == 3:  # Calibrate LEFT
                if len(calibration_left_samples) < 60:
                    calibration_left_samples.append(smoothed_horizontal)
                if len(calibration_distance_samples) < 40:
                    calibration_distance_samples.append(distanceCM)  
                cv2.putText(frame, "CALIBRATION: Look LEFT", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
                cv2.putText(frame, f"Samples: {len(calibration_left_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_left_samples) == 60:
                    calibration_horizontal_left, is_left_calibrated = calculate_mean(calibration_left_samples)
                    print(f"LEFT Calibrated! Position: {calibration_horizontal_left:.3f}")
                    cv2.putText(frame, "Press 'c' to continue to CENTER calibration", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        calibration_stage = 4
            
            elif calibration_stage == 4:  # Calibrate CENTER (horizontal)
                if len(calibration_h_center_samples) < 60:
                    calibration_h_center_samples.append(smoothed_horizontal)
                if len(calibration_distance_samples) < 50:
                    calibration_distance_samples.append(distanceCM)  
                cv2.putText(frame, "CALIBRATION: Look CENTER", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3)
                cv2.putText(frame, f"Samples: {len(calibration_h_center_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                if len(calibration_h_center_samples) == 60:
                    calibration_horizontal_center, is_h_center_calibrated = calculate_mean(calibration_h_center_samples)
                    print(f"H-CENTER Calibrated! Position: {calibration_horizontal_center:.3f}")
                    cv2.putText(frame, "Press 'c' to continue to RIGHT calibration", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        calibration_stage = 5
            
            elif calibration_stage == 5:  # Calibrate RIGHT
                if len(calibration_right_samples) < 60:
                    calibration_right_samples.append(smoothed_horizontal)
                if len(calibration_distance_samples) < 60:
                    calibration_distance_samples.append(distanceCM) 
                cv2.putText(frame, "CALIBRATION: Look RIGHT", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 3)
                cv2.putText(frame, f"Samples: {len(calibration_right_samples)}/60", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                if len(calibration_distance_samples) == 60:
                    distance_threshold, _ = calculate_mean(calibration_distance_samples)
                if len(calibration_right_samples) == 60:
                    calibration_horizontal_right, is_right_calibrated = calculate_mean(calibration_right_samples)
                    # Calculate thresholds as midpoints
                    left_threshold = (calibration_horizontal_left + calibration_horizontal_center) / 2
                    right_threshold = (calibration_horizontal_center + calibration_horizontal_right) / 2
                    cv2.putText(frame, "Calibration complete! Press 's' to SAVE, 'c' to start tracking.", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('s'):
                        # Calculate distance threshold before saving
                        save_calibration()
                        # Put text confirmation
                        cv2.putText(frame, "Calibration data SAVED!", (frame.shape[1]-400, frame.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    elif key == ord('c'):
                        calibration_stage = 6
                    
            
            else:  # calibration_stage == 6 (tracking mode)
                # Calculate distance threshold
                    
                if distanceCM > (distance_threshold + 10):
                    distance_status = "FAR"
                else:
                    distance_status = "GOOD"

                # Determine vertical gaze (up/down)
                # First check iris-based vertical gaze
                if smoothed_height > up_threshold:
                    eye_vertical_gaze = "UP"
                    v_color = (0, 255, 0)
                elif smoothed_height < down_threshold:
                    eye_vertical_gaze = "DOWN"
                    v_color = (0, 0, 255)
                else:
                    eye_vertical_gaze = "CENTER"
                    v_color = (255, 255, 0)

                # Determine horizontal gaze (left/right)
                if smoothed_horizontal < left_threshold:
                    eye_horizontal_gaze = "LEFT"
                    h_color = (255, 0, 255)  # Magenta
                elif smoothed_horizontal > right_threshold:
                    eye_horizontal_gaze = "RIGHT"
                    h_color = (0, 165, 255)  # Orange
                else:
                    eye_horizontal_gaze = "CENTER"
                    h_color = (255, 255, 0)

                # Now also check face gaze position (smoothed)
                face_vertical_gaze = "CENTER"
                face_horizontal_gaze = "CENTER"

                # Define screen thirds for face gaze
                third_h = SCREEN_HEIGHT // 3
                two_thirds_h = 2 * third_h
                third_w = SCREEN_WIDTH // 3
                two_thirds_w = 2 * third_w

                # Determine face-based vertical gaze
                if face_gaze_position[1] < third_h:
                    face_vertical_gaze = "UP"
                elif face_gaze_position[1] > two_thirds_h:
                    face_vertical_gaze = "DOWN"

                # Determine face-based horizontal gaze
                if face_gaze_position[0] < third_w:
                    face_horizontal_gaze = "LEFT"
                elif face_gaze_position[0] > two_thirds_w:
                    face_horizontal_gaze = "RIGHT"

                # Combine iris and face gaze (give priority to agreement between both)
                # If both agree, use that direction. If they disagree, use iris as primary
                final_vertical_gaze = eye_vertical_gaze
                final_horizontal_gaze = eye_horizontal_gaze

                # Override with face gaze if iris is CENTER but face detects direction
                if eye_vertical_gaze == "UP" and face_vertical_gaze == "CENTER":
                    final_vertical_gaze = "UP"
                elif eye_vertical_gaze == "UP" and face_vertical_gaze == "UP":
                    final_vertical_gaze = "UP"
                elif eye_vertical_gaze == "DOWN" and face_vertical_gaze == "CENTER":
                    final_vertical_gaze = "DOWN"
                elif eye_vertical_gaze == "DOWN" and face_vertical_gaze == "DOWN":
                    final_vertical_gaze = "DOWN"
                if eye_horizontal_gaze == "LEFT" and face_horizontal_gaze == "CENTER":
                    final_horizontal_gaze = "LEFT"
                elif eye_horizontal_gaze == "LEFT" and face_horizontal_gaze == "LEFT":
                    final_horizontal_gaze = "LEFT"
                elif eye_horizontal_gaze == "RIGHT" and face_horizontal_gaze == "CENTER":
                    final_horizontal_gaze = "RIGHT"
                elif eye_horizontal_gaze == "RIGHT" and face_horizontal_gaze == "RIGHT":
                    final_horizontal_gaze = "RIGHT"
                elif eye_horizontal_gaze == "RIGHT" and face_horizontal_gaze == "LEFT":
                    final_horizontal_gaze = "CENTER"
                elif eye_horizontal_gaze == "LEFT" and face_horizontal_gaze == "RIGHT":
                    final_horizontal_gaze = "CENTER"

                # Combined gaze direction
                if final_vertical_gaze == "CENTER" and final_horizontal_gaze == "CENTER":
                    combined_gaze = "CENTER"
                    combined_color = (255, 255, 255)
                else:
                    combined_gaze = f"{final_vertical_gaze} {final_horizontal_gaze}"
                    combined_color = (0, 255, 255)

                # Display gaze directions
                cv2.putText(frame, combined_gaze, 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, combined_color, 4)

                cv2.putText(frame, f"V: {final_vertical_gaze}", 
                           (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, v_color, 2)
                cv2.putText(frame, f"H: {final_horizontal_gaze}", 
                           (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, h_color, 2)

                #Display distance status
                cv2.putText(frame, f"Distance: {distance_status}", 
                           (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0) if distance_status == "GOOD" else (0, 0, 255), 2)  

                # Display metrics with calibration points
                cv2.putText(frame, f"Height: {smoothed_height:.1f} (U: {calibration_height_up:.0f} C:{calibration_height_center:.0f} D:{calibration_height_down:.0f})", 
                           (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(frame, f"H-Pos: {smoothed_horizontal:.3f} (L:{calibration_horizontal_left:.2f} C:{calibration_horizontal_center:.2f} R:{calibration_horizontal_right:.2f})", 
                           (50, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Visual indicator bars
                v_bar_x = 50
                v_bar_y = 350
                bar_width = 400
                bar_height = 20
                
                # Vertical bar
                cv2.rectangle(frame, (v_bar_x, v_bar_y), (v_bar_x + bar_width, v_bar_y + bar_height), 
                             (100, 100, 100), 2)
                
                # Normalize to 0-1 range based on calibration
                v_range = calibration_height_up - calibration_height_down
                if v_range > 0:
                    v_norm = (smoothed_height - calibration_height_down) / v_range
                    v_fill = int(v_norm * bar_width)
                    v_fill = max(0, min(bar_width, v_fill))
                    cv2.rectangle(frame, (v_bar_x, v_bar_y), (v_bar_x + v_fill, v_bar_y + bar_height), 
                                 v_color, -1)
                
                # Mark calibration points on vertical bar
                up_mark = int(((calibration_height_up - calibration_height_down) / v_range) * bar_width) if v_range > 0 else bar_width
                center_mark = int(((calibration_height_center - calibration_height_down) / v_range) * bar_width) if v_range > 0 else bar_width // 2
                cv2.line(frame, (v_bar_x + up_mark, v_bar_y), (v_bar_x + up_mark, v_bar_y + bar_height), (0, 255, 0), 2)
                cv2.line(frame, (v_bar_x + center_mark, v_bar_y), (v_bar_x + center_mark, v_bar_y + bar_height), (255, 255, 0), 2)
                cv2.line(frame, (v_bar_x, v_bar_y), (v_bar_x, v_bar_y + bar_height), (0, 0, 255), 2)
                
                # Horizontal bar
                h_bar_y = 380
                cv2.rectangle(frame, (v_bar_x, h_bar_y), (v_bar_x + bar_width, h_bar_y + bar_height), 
                             (100, 100, 100), 2)
                
                # Normalize to 0-1 range based on calibration
                h_range = calibration_horizontal_right - calibration_horizontal_left
                if h_range > 0:
                    h_norm = (smoothed_horizontal - calibration_horizontal_left) / h_range
                    h_fill = int(h_norm * bar_width)
                    h_fill = max(0, min(bar_width, h_fill))
                    cv2.rectangle(frame, (v_bar_x, h_bar_y), (v_bar_x + h_fill, h_bar_y + bar_height), 
                                 h_color, -1)
                
                # Mark calibration points on horizontal bar
                left_mark = 0
                center_mark = int(((calibration_horizontal_center - calibration_horizontal_left) / h_range) * bar_width) if h_range > 0 else bar_width // 2
                right_mark = bar_width
                cv2.line(frame, (v_bar_x + left_mark, h_bar_y), (v_bar_x + left_mark, h_bar_y + bar_height), (255, 0, 255), 2)
                cv2.line(frame, (v_bar_x + center_mark, h_bar_y), (v_bar_x + center_mark, h_bar_y + bar_height), (255, 255, 0), 2)
                cv2.line(frame, (v_bar_x + right_mark, h_bar_y), (v_bar_x + right_mark, h_bar_y + bar_height), (0, 165, 255), 2)
            
            #endregion Calibration & Gaze Detection

            # ============================================ CALIBRATION & GAZE DETECTION =============================================== #

            # Screen quadrants visualization
            if calibration_stage == 6:
                screen_frame = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                
                # Calculate thirds for dynamic scaling
                third_w = SCREEN_WIDTH // 3
                two_thirds_w = 2 * third_w
                third_h = SCREEN_HEIGHT // 3
                two_thirds_h = 2 * third_h
                
                if combined_gaze == "UP LEFT":
                    segment_rect_dimensions = (0, 0, SCREEN_WIDTH // 2, third_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 255, 0), -1)
                elif combined_gaze == "UP RIGHT":
                    segment_rect_dimensions = (SCREEN_WIDTH // 2, 0, SCREEN_WIDTH, third_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 255, 0), -1)
                elif combined_gaze == "DOWN LEFT":
                    segment_rect_dimensions = (0, two_thirds_h, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 0, 255), -1)
                elif combined_gaze == "DOWN RIGHT":
                    segment_rect_dimensions = (SCREEN_WIDTH // 2, two_thirds_h, SCREEN_WIDTH, SCREEN_HEIGHT)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 0, 255), -1)
                elif combined_gaze == "UP CENTER":
                    segment_rect_dimensions = (third_w, 0, two_thirds_w, third_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 255, 128), -1)
                elif combined_gaze == "DOWN CENTER":
                    segment_rect_dimensions = (third_w, two_thirds_h, two_thirds_w, SCREEN_HEIGHT)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (0, 128, 255), -1)
                elif combined_gaze == "CENTER LEFT":
                    segment_rect_dimensions = (0, third_h, SCREEN_WIDTH // 2, two_thirds_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (255, 128, 0), -1)
                elif combined_gaze == "CENTER RIGHT":
                    segment_rect_dimensions = (SCREEN_WIDTH // 2, third_h, SCREEN_WIDTH, two_thirds_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (128, 0, 255), -1)
                elif combined_gaze == "CENTER":
                    segment_rect_dimensions = (third_w, third_h, two_thirds_w, two_thirds_h)
                    cv2.rectangle(screen_frame, (segment_rect_dimensions[0], segment_rect_dimensions[1]), 
                                (segment_rect_dimensions[2], segment_rect_dimensions[3]), (255, 255, 255), -1)

                screen_frame = cv2.resize(screen_frame, (600, 400))
                cv2.imshow('Screen Quadrants', screen_frame)

            # ============================================= GAZE MAPPING BY IRIS =============================================== #

            # Calculate iris position within eye rectangles (normalized 0-1)

            # Right eye
            re_width = abs(ef_re_x2 - ef_re_x1)
            re_height = abs(ef_re_y2 - ef_re_y1)
            re_iris_x_norm = (right_eye_center[0] - ef_re_x1) / re_width if re_width > 0 else 0.5
            re_iris_y_norm = (right_eye_center[1] - ef_re_y1) / re_height if re_height > 0 else 0.5

            # Left eye
            le_width = abs(ef_le_x2 - ef_le_x1)
            le_height = abs(ef_le_y2 - ef_le_y1)
            le_iris_x_norm = (left_eye_center[0] - ef_le_x1) / le_width if le_width > 0 else 0.5
            le_iris_y_norm = (left_eye_center[1] - ef_le_y1) / le_height if le_height > 0 else 0.5

            # Average both eyes for stability
            avg_iris_x_norm = (re_iris_x_norm + le_iris_x_norm) / 2
            avg_iris_y_norm = (re_iris_y_norm + le_iris_y_norm) / 2

            # Only map gaze position if calibration is complete
            if calibration_stage == 6:
                # Calculate dynamic thirds based on screen dimensions
                third_w = SCREEN_WIDTH // 3
                two_thirds_w = 2 * third_w
                third_h = SCREEN_HEIGHT // 3
                two_thirds_h = 2 * third_h
                
                # Map normalized iris position to calibrated screen coordinates
                # Horizontal mapping (left to right)
                if avg_iris_x_norm < left_threshold:
                    # Looking left
                    # Map from [calibration_horizontal_left to left_threshold] -> [0 to third_w]
                    h_range = left_threshold - calibration_horizontal_left
                    if h_range > 0:
                        norm_in_range = (avg_iris_x_norm - calibration_horizontal_left) / h_range
                        gaze_x = int(norm_in_range * third_w)  # Left third of screen
                    else:
                        gaze_x = third_w // 2
                elif avg_iris_x_norm > right_threshold:
                    # Looking right
                    # Map from [right_threshold to calibration_horizontal_right] -> [two_thirds_w to SCREEN_WIDTH]
                    h_range = calibration_horizontal_right - right_threshold
                    if h_range > 0:
                        norm_in_range = (avg_iris_x_norm - right_threshold) / h_range
                        gaze_x = int(two_thirds_w + norm_in_range * third_w)  # Right third of screen
                    else:
                        gaze_x = two_thirds_w + third_w // 2
                else:
                    # Looking center
                    # Map from [left_threshold to right_threshold] -> [third_w to two_thirds_w]
                    h_range = right_threshold - left_threshold
                    if h_range > 0:
                        norm_in_range = (avg_iris_x_norm - left_threshold) / h_range
                        gaze_x = int(third_w + norm_in_range * third_w)  # Center third of screen
                    else:
                        gaze_x = SCREEN_WIDTH // 2

                # Vertical mapping (up to down)
                if avg_rect_height > up_threshold:
                    # Looking up
                    # Map from [up_threshold to calibration_height_up] -> [0 to third_h]
                    v_range = calibration_height_up - up_threshold
                    if v_range > 0:
                        norm_in_range = (avg_rect_height - up_threshold) / v_range
                        gaze_y = int(third_h - norm_in_range * third_h)  # Top third (inverted)
                    else:
                        gaze_y = third_h // 2
                elif avg_rect_height < down_threshold:
                    # Looking down
                    # Map from [calibration_height_down to down_threshold] -> [two_thirds_h to SCREEN_HEIGHT]
                    v_range = down_threshold - calibration_height_down
                    if v_range > 0:
                        norm_in_range = (avg_rect_height - calibration_height_down) / v_range
                        gaze_y = int(two_thirds_h + norm_in_range * third_h)  # Bottom third
                    else:
                        gaze_y = two_thirds_h + third_h // 2
                else:
                    # Looking center
                    # Map from [down_threshold to up_threshold] -> [third_h to two_thirds_h]
                    v_range = up_threshold - down_threshold
                    if v_range > 0:
                        norm_in_range = (avg_rect_height - down_threshold) / v_range
                        gaze_y = int(third_h + norm_in_range * third_h)  # Center third
                    else:
                        gaze_y = SCREEN_HEIGHT // 2

                # # Clamp to screen bounds
                # gaze_x = max(0, min(SCREEN_WIDTH, gaze_x))
                # gaze_y = max(0, min(SCREEN_HEIGHT, gaze_y))

                rect = segment_rect_dimensions
                x1, y1, x2, y2 = rect
                seg_w = x2 - x1
                seg_h = y2 - y1

                # Normalize gaze_x, gaze_y to [0,1] based on the full screen
                norm_x = gaze_x / (SCREEN_WIDTH + (SCREEN_HEIGHT * 0.25))
                norm_y = gaze_y / (SCREEN_HEIGHT + (SCREEN_WIDTH * 0.25))

                # Map normalized gaze to segment
                mapped_x = int(x1 + norm_x * seg_w)
                mapped_y = int(y1 + norm_y * seg_h)

                # Clamp to segment bounds
                mapped_x = max(x1, min(x2 - 1, mapped_x))
                mapped_y = max(y1, min(y2 - 1, mapped_y))
                
                # Update gaze position
                iris_gaze_position = (mapped_x, mapped_y)
                iris_gaze_position_history.append(iris_gaze_position)

                if len(iris_gaze_position_history) > 0 :
                    # Smooth gaze position
                    smoothed_x = int(np.mean([pos[0] for pos in iris_gaze_position_history] + [iris_gaze_position[0]]))
                    smoothed_y = int(np.mean([pos[1] for pos in iris_gaze_position_history] + [iris_gaze_position[1]]))
                    iris_gaze_position = (smoothed_x, smoothed_y)
                
                # Display normalized iris positions on eye frame
                cv2.putText(eye_frame, f"Iris X: {avg_iris_x_norm:.2f}", 
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(eye_frame, f"Iris Y: {avg_iris_y_norm:.2f}", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(eye_frame, f"Gaze: ({gaze_x}, {gaze_y})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
    # ============================================= GAZE MAPPING BY IRIS =============================================== #

    
    
    # Gaze position display
    if calibration_stage == 6:

        gaze_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # draw circle at face-based gaze position
        cv2.circle(gaze_frame, face_gaze_position, 15, (255, 0, 0), -1)

        # draw circle at eye-based gaze position
        cv2.circle(gaze_frame, iris_gaze_position, 15, (0, 0, 255), -1)

        # Weighted average of the two gaze positions
        # Adjust weights as needed: higher weight = more influence
        face_weight = 0.50  # 50% face-based tracking
        eye_weight = 0.50 # 50% eye-based tracking
        
        weighted_gaze_x = int((face_gaze_position[0] * face_weight + iris_gaze_position[0] * eye_weight))
        weighted_gaze_y = int((face_gaze_position[1] * face_weight + iris_gaze_position[1] * eye_weight))

        # 80px circle outline for weighted gaze 
        cv2.circle(gaze_frame, (weighted_gaze_x, weighted_gaze_y), 100, (0, 255, 255), 3)
       
        # Peripheral circles for reference
        cv2.circle(gaze_frame, (weighted_gaze_x, weighted_gaze_y), 200, (255, 255, 255), 2)
        cv2.circle(gaze_frame, (weighted_gaze_x, weighted_gaze_y), 300, (255, 255, 255), 1)

        # Display legend following circles
        cv2.putText(gaze_frame, "Face Gaze (Blue)", (face_gaze_position[0], face_gaze_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(gaze_frame, "Eye Gaze (Red)", (iris_gaze_position[0], iris_gaze_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(gaze_frame, "Focus Gaze (Yellow)", (weighted_gaze_x, weighted_gaze_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        cv2.imshow('Gaze Position', gaze_frame)

    frame = cv2.resize(frame, (854, 480))
    cv2.imshow('Eye Tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and calibration_stage == 6:
        # Full recalibration
        calibration_stage = -1
        calibration_up_samples.clear()
        calibration_center_samples.clear()
        calibration_down_samples.clear()
        calibration_left_samples.clear()
        calibration_h_center_samples.clear()
        calibration_right_samples.clear()
        height_history.clear()
        horizontal_history.clear()
        is_up_calibrated = False
        is_center_calibrated = False
        is_down_calibrated = False
        is_left_calibrated = False
        is_h_center_calibrated = False
        is_right_calibrated = False
        print("Recalibrating all axes...")
    elif key == ord('s') and calibration_stage == 6:
        # Save calibration anytime during tracking
        save_calibration()
    elif key == ord('l'):
        # Load calibration
        if load_calibration():
            print("Calibration loaded successfully!")
        else:
            print("Failed to load calibration or file not found")

cam.release()
cv2.destroyAllWindows()