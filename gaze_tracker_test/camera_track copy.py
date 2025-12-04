import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque
import json
import os
import keyboard

try:
    from .camera_track_constants import GazeTrackerState
except ImportError:
    from camera_track_constants import GazeTrackerState

class CameraTrack:
    def __init__(self, camera_index=0, screen_width=1920, screen_height=1080):
        self.cam = cv2.VideoCapture(camera_index)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                           max_num_faces=1,
                                                           refine_landmarks=True,
                                                           min_detection_confidence=0.5,
                                                           min_tracking_confidence=0.5)
        
         # Screen dimensions for gaze mapping
        if not self.cam.isOpened():
            raise ValueError("Could not open camera.")
        
        self.track_constants = GazeTrackerState()
         # Ray smoothing
        self.ray_origins = deque(maxlen=5)
        self.ray_directions = deque(maxlen=5)
        self.ray_length = 50

        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        
        # Gaze positions
        self.iris_gaze_position = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        self.face_gaze_position_history = deque(maxlen=5)
        self.iris_gaze_position_history = deque(maxlen=5)
        self.combined_color_history = deque(maxlen=5)
        
        # Calibration samples
        self.calibration_distance_samples = deque(maxlen=60)
        self.calibration_left_samples = deque(maxlen=60)
        self.calibration_h_center_samples = deque(maxlen=60)
        self.calibration_right_samples = deque(maxlen=60)
        self.calibration_up_samples = deque(maxlen=60)
        self.calibration_center_samples = deque(maxlen=60)
        self.calibration_down_samples = deque(maxlen=60)
        
        # Calibration values
        self.calibration_height_up = 0.0
        self.calibration_height_center = 0.0
        self.calibration_height_down = 0.0
        self.calibration_horizontal_left = 0.0
        self.calibration_horizontal_center = 0.0
        self.calibration_horizontal_right = 0.0
        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0
        
        # Calibration flags
        self.is_up_calibrated = False
        self.is_center_calibrated = False
        self.is_down_calibrated = False
        self.is_left_calibrated = False
        self.is_h_center_calibrated = False
        self.is_right_calibrated = False
        
        # History deques
        self.height_history = deque(maxlen=5)
        self.horizontal_history = deque(maxlen=5)
        
        # Gaze tracking
        self.combined_gaze = None
        
        # Thresholds
        self.up_threshold = 0
        self.down_threshold = 0
        self.left_threshold = 0
        self.right_threshold = 0
        self.distance_threshold = 0
        
        # Calibration stage
        self.calibration_stage = -1

        # Weighted Screen Position
        self.weighted_screen_position = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2)
        
    
    # Calculate bounding box around both eyesq
    def get_eye_bbox(self, all_eye_points, w, h, padding=5):
        all_eye_points = np.array(all_eye_points)
        x_min = max(0, int(all_eye_points[:, 0].min()) - padding - 20)
        x_max = min(w, int(all_eye_points[:, 0].max()) + padding + 20)
        y_min = max(0, int(all_eye_points[:, 1].min()) - padding - 10)
        y_max = min(h, int(all_eye_points[:, 1].max()) + padding + 10)
        return x_min, x_max, y_min, y_max


    # Draw eye landmarks and return points and frame
    def draw_eye_landmarks(self, landmarks, indices, w, h, frame):
                    all_eye_points = []
                    all_eye_idx = []
                    for idx in indices:
                        landmark = landmarks[idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        all_eye_points.append((x, y))
                        all_eye_idx.append(idx)
                        # Debugging: draw all eye landmarks
                        # cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
                    return all_eye_points, all_eye_idx, frame

        # JSON Calibration Functions
    def save_calibration(self):
        """Save calibration data to JSON file"""
        calibration_data = {
            'calibration_height_up': float(self.calibration_height_up),
            'calibration_height_center': float(self.calibration_height_center),
            'calibration_height_down': float(self.calibration_height_down),
            'calibration_horizontal_left': float(self.calibration_horizontal_left),
            'calibration_horizontal_center': float(self.calibration_horizontal_center),
            'calibration_horizontal_right': float(self.calibration_horizontal_right),
            'calibration_offset_yaw': float(self.calibration_offset_yaw),
            'calibration_offset_pitch': float(self.calibration_offset_pitch),
            'up_threshold': float(self.up_threshold),
            'down_threshold': float(self.down_threshold),
            'left_threshold': float(self.left_threshold),
            'right_threshold': float(self.right_threshold),
            'distance_threshold': float(self.distance_threshold),
            'is_up_calibrated': self.is_up_calibrated,
            'is_center_calibrated': self.is_center_calibrated,
            'is_down_calibrated': self.is_down_calibrated,
            'is_left_calibrated': self.is_left_calibrated,
            'is_h_center_calibrated': self.is_h_center_calibrated,
            'is_right_calibrated': self.is_right_calibrated
        }
        
        return calibration_data
        
    def load_calibration(self,calibration_data):
        """Load calibration data from JSON file"""
        if calibration_data:
            
            self.calibration_height_up = calibration_data['calibration_height_up']
            self.calibration_height_center = calibration_data['calibration_height_center']
            self.calibration_height_down = calibration_data['calibration_height_down']
            self.calibration_horizontal_left = calibration_data['calibration_horizontal_left']
            self.calibration_horizontal_center = calibration_data['calibration_horizontal_center']
            self.calibration_horizontal_right = calibration_data['calibration_horizontal_right']
            self.calibration_offset_yaw = calibration_data['calibration_offset_yaw']
            self.calibration_offset_pitch = calibration_data['calibration_offset_pitch']
            self.up_threshold = calibration_data['up_threshold']
            self.down_threshold = calibration_data['down_threshold']
            self.left_threshold = calibration_data['left_threshold']
            self.right_threshold = calibration_data['right_threshold']
            self.distance_threshold = calibration_data['distance_threshold']
            self.is_up_calibrated = calibration_data['is_up_calibrated']
            self.is_center_calibrated = calibration_data['is_center_calibrated']
            self.is_down_calibrated = calibration_data['is_down_calibrated']
            self.is_left_calibrated = calibration_data['is_left_calibrated']
            self.is_h_center_calibrated = calibration_data['is_h_center_calibrated']
            self.is_right_calibrated = calibration_data['is_right_calibrated']
            
            # Set calibration stage to tracking mode
            self.calibration_stage = 6
        
            

        
    def update_calibration_stage(self, stage):
        if stage > 6:
            stage = 6
        elif stage < -1:
            stage = -1
        else:
            self.calibration_stage = stage

    def reset_calibration(self):
        """Reset all calibration values to default"""
        self.calibration_stage = -1
        self.is_up_calibrated = False
        self.is_center_calibrated = False
        self.is_down_calibrated = False
        self.is_left_calibrated = False
        self.is_h_center_calibrated = False
        self.is_right_calibrated = False
        self.calibration_height_up = 0.0
        self.calibration_height_center = 0.0
        self.calibration_height_down = 0.0
        self.calibration_horizontal_left = 0.0
        self.calibration_horizontal_center = 0.0
        self.calibration_horizontal_right = 0.0
        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0
        self.up_threshold = 0    
        self.down_threshold = 0
        self.left_threshold = 0
        self.right_threshold = 0
        self.distance_threshold = 0
           
        # Clear calibration samples
        self.calibration_distance_samples.clear()
        self.calibration_left_samples.clear()
        self.calibration_h_center_samples.clear()
        self.calibration_right_samples.clear()
        self.calibration_up_samples.clear()
        self.calibration_center_samples.clear()
        self.calibration_down_samples.clear()

    # Find mean of calibration samples
    def calculate_mean(self, samples):
        return (np.mean(samples), True) if samples else (0, False)

    def calculate_pts_mean(self, pts):
        return np.mean(pts, axis=0).astype(int)

    # Calculate iris centers in original frame coordinates
    def get_iris_center(self, landmarks, indices, w, h):
        x = np.mean([landmarks[idx].x for idx in indices]) * w
        y = np.mean([landmarks[idx].y for idx in indices]) * h
        return x, y

    def project(self, pt3d):
                return int(pt3d[0]), int(pt3d[1])
    
    def get_frame(self):
        # Unpack constants
        
        # left and right eyelid landmark indices
        left_eye_indices = self.track_constants.left_eye_indices
        right_eye_indices = self.track_constants.right_eye_indices

        face_indices = self.track_constants.face_indices
        padding = self.track_constants.padding

        # y width in centimeters
        y_dist = self.track_constants.y_dist
        cm_dist = self.track_constants.cm_dist
        dist_coff = self.track_constants.dist_coff

        # Face Axis Key Indexes
        KEY_FACE_LANDMARKS = self.track_constants.KEY_FACE_LANDMARKS

        # ========================== STATE & CALIBRATION VARIABLES ==========================

        # Ray smoothing for head pose
        ray_origins = self.ray_origins
        ray_directions = self.ray_directions
        ray_length = self.ray_length

        # Gaze Circle Position
        iris_gaze_position = self.iris_gaze_position

        # Utility deques for smoothing gaze positions and colors
        face_gaze_position_history = self.face_gaze_position_history
        iris_gaze_position_history = self.iris_gaze_position_history
        combined_color_history = self.combined_color_history

        # Calibration samples deques
        calibration_distance_samples = self.calibration_distance_samples
        calibration_left_samples = self.calibration_left_samples
        calibration_h_center_samples = self.calibration_h_center_samples
        calibration_right_samples = self.calibration_right_samples
        calibration_up_samples = self.calibration_up_samples
        calibration_center_samples = self.calibration_center_samples
        calibration_down_samples = self.calibration_down_samples

        # Calibration and smoothing for rectangle height (vertical)
        calibration_height_up = self.calibration_height_up
        calibration_height_center = self.calibration_height_center
        calibration_height_down = self.calibration_height_down

        is_up_calibrated = self.is_up_calibrated
        is_center_calibrated = self.is_center_calibrated
        is_down_calibrated = self.is_down_calibrated
        height_history = self.height_history

        # Calibration for horizontal (left/right)
        calibration_horizontal_left = self.calibration_horizontal_left
        calibration_horizontal_center = self.calibration_horizontal_center
        calibration_horizontal_right = self.calibration_horizontal_right

        calibration_offset_yaw = self.calibration_offset_yaw
        calibration_offset_pitch = self.calibration_offset_pitch

        is_left_calibrated = self.is_left_calibrated
        is_h_center_calibrated = self.is_h_center_calibrated
        is_right_calibrated = self.is_right_calibrated
        horizontal_history = self.horizontal_history

        # Combined gaze direction string
        combined_gaze = self.combined_gaze

        # Thresholds (will be calculated from calibration data)
        up_threshold = self.up_threshold
        down_threshold = self.down_threshold
        left_threshold = self.left_threshold
        right_threshold = self.right_threshold
        distance_threshold = self.distance_threshold
        calibration_stage = self.calibration_stage
        
        # Read frame from camera
        ret, frame = self.cam.read()
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        landmarks_points = results.multi_face_landmarks

        if landmarks_points:       
                landmarks = landmarks_points[0].landmark
                all_eye_points = []
                all_eye_idx = []
                right_eye_points = []
                left_eye_points = []
                
                # Right eye: landmarks 474-477
                right_eye_points, right_eye_idx, frame = self.draw_eye_landmarks(landmarks, range(473, 478), w, h, frame)
                
                # Left eye: landmarks 469-472
                left_eye_points, left_eye_idx, frame = self.draw_eye_landmarks(landmarks, range(468, 473), w, h, frame)
                    
                # Draw all eye landmarks
                all_eye_points, all_eye_idx, frame = self.draw_eye_landmarks(landmarks, left_eye_indices + right_eye_indices, w, h, frame)


                # ============================================= FACE AXIS ALGORITHMS =============================================== #

                # region FACE AXIS ALGORITHMS

                def landmark_to_np(landmark, w, h):
                    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

                key_points = {}
                for name, idx in KEY_FACE_LANDMARKS.items():
                    pt = landmark_to_np(landmarks[idx], w, h)
                    key_points[name] = pt
                    x, y = int(pt[0]), int(pt[1])
                    
                    # Debugging: draw key face landmarks
                    # cv2.circle(frame, (x,y), 10, (0,0,255), -1)
                
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

                # Debugging: write width_pts in frame
                # cv2.putText(frame, f"Distance CM: {distanceCM:.2f}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)    

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
                avg_origin_x, avg_origin_y = self.project(avg_origin)
                ray_end_x, ray_end_y = self.project(ray_end)

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

                screen_frame_w, screen_frame_h = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
                
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
                
                # ============================================= EYE FRAME DRAWING =============================================== #

                # Calculate bounding box around both eyes
                all_eye_points = np.array(all_eye_points)
                x_min, x_max, y_min, y_max = self.get_eye_bbox(all_eye_points, w, h, padding)

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
                left_iris_center_x, left_iris_center_y = self.get_iris_center(landmarks, range(468, 473), w, h)
                right_iris_center_x, right_iris_center_y = self.get_iris_center(landmarks, range(473, 478), w, h)

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

                # region DEBUG DRAWINGS IN EYE FRAME
                # # Drawings in eye_frame for debugging
                # # Convert to eye_frame coordinates
                # ef_ray_end = ray_end - np.array([x_min, y_min, 0])
                # ef_avg_origin = avg_origin - np.array([x_min, y_min, 0])

                # # Draw ray in eye_frame
                # cv2.line(eye_frame, self.project(ef_avg_origin), self.project(ef_ray_end), (15, 255, 0), 3)

                # endregion

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

                # # Draw iris centers
                right_eye_center = np.mean(right_eye_points, axis=0).astype(int)

                left_eye_center = np.mean(left_eye_points, axis=0).astype(int)

                # ============================================= EYE FRAME DRAWING =============================================== #
                # ============================================ CALIBRATION & GAZE DETECTION =============================================== #

                #region Calibration & Gaze Detection

                # Multi-stage calibration
                if calibration_stage == -1:
                    cv2.putText(frame, "Press 'c' to start CALIBRATION for UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
                    
                    
                if calibration_stage == 0:  # Calibrate UP
                    calibration_offset_yaw = 180 - raw_yaw_deg
                    calibration_offset_pitch = 180 - raw_pitch_deg

                    if len(calibration_up_samples) < 60:
                        calibration_up_samples.append(smoothed_height)
                    if len(calibration_distance_samples) < 10:
                        calibration_distance_samples.append(distanceCM)                
                    
                    if len(calibration_up_samples) == 60:
                        calibration_height_up, is_up_calibrated = self.calculate_mean(calibration_up_samples)
                
                elif calibration_stage == 1:  # Calibrate CENTER (vertical)
                    

                    if len(calibration_center_samples) < 60:
                        calibration_center_samples.append(smoothed_height)
                    if len(calibration_distance_samples) < 20:
                        calibration_distance_samples.append(distanceCM)  
                    
                    if len(calibration_center_samples) == 60:
                        calibration_height_center, is_center_calibrated = self.calculate_mean(calibration_center_samples)
                
                elif calibration_stage == 2:  # Calibrate DOWN
                    if len(calibration_down_samples) < 60:
                        calibration_down_samples.append(smoothed_height)
                    if len(calibration_distance_samples) < 30:
                        calibration_distance_samples.append(distanceCM)  

                    
                    if len(calibration_down_samples) == 60:
                        calibration_height_down, is_down_calibrated = self.calculate_mean(calibration_down_samples)
                        # Calculate thresholds as midpoints
                        up_threshold = (calibration_height_up + calibration_height_center) / 2
                        down_threshold = (calibration_height_center + calibration_height_down) / 2

                            
                
                elif calibration_stage == 3:  # Calibrate LEFT
                    if len(calibration_left_samples) < 60:
                        calibration_left_samples.append(smoothed_horizontal)
                    if len(calibration_distance_samples) < 40:
                        calibration_distance_samples.append(distanceCM)  
                    
                    if len(calibration_left_samples) == 60:
                        calibration_horizontal_left, is_left_calibrated = self.calculate_mean(calibration_left_samples)
                
                elif calibration_stage == 4:  # Calibrate CENTER (horizontal)
                    if len(calibration_h_center_samples) < 60:
                        calibration_h_center_samples.append(smoothed_horizontal)
                    if len(calibration_distance_samples) < 50:
                        calibration_distance_samples.append(distanceCM)  
                    
                    if len(calibration_h_center_samples) == 60:
                        calibration_horizontal_center, is_h_center_calibrated = self.calculate_mean(calibration_h_center_samples)

                elif calibration_stage == 5:  # Calibrate RIGHT
                    if len(calibration_right_samples) < 60:
                        calibration_right_samples.append(smoothed_horizontal)
                    if len(calibration_distance_samples) < 60:
                        calibration_distance_samples.append(distanceCM) 

                    if len(calibration_distance_samples) == 60:
                        distance_threshold, _ = self.calculate_mean(calibration_distance_samples)
                    if len(calibration_right_samples) == 60:
                        calibration_horizontal_right, is_right_calibrated = self.calculate_mean(calibration_right_samples)
                        # Calculate thresholds as midpoints
                        left_threshold = (calibration_horizontal_left + calibration_horizontal_center) / 2
                        right_threshold = (calibration_horizontal_center + calibration_horizontal_right) / 2
                
                else:  # calibration_stage == 6 (tracking mode)
                    # Calculate distance threshold
                    
                    if distanceCM > (distance_threshold + 10):
                        distance_status = "FAR"
                    else:
                        distance_status = "GOOD"

                    if smoothed_height > up_threshold:
                        eye_vertical_gaze = "UP"
                        v_color = (0, 255, 0)
                        eye_vertical_margin = smoothed_height - up_threshold
                    elif smoothed_height < down_threshold:
                        eye_vertical_gaze = "DOWN"
                        v_color = (0, 0, 255)
                        eye_vertical_margin = down_threshold - smoothed_height
                    else:
                        eye_vertical_gaze = "CENTER"
                        v_color = (255, 255, 0)
                        # Distance from nearest threshold
                        eye_vertical_margin = min(abs(smoothed_height - up_threshold), abs(smoothed_height - down_threshold))

                    # Determine horizontal gaze (left/right)
                    if smoothed_horizontal < left_threshold:
                        eye_horizontal_gaze = "LEFT"
                        h_color = (255, 0, 255)  # Magenta
                        eye_horizontal_margin = left_threshold - smoothed_horizontal
                    elif smoothed_horizontal > right_threshold:
                        eye_horizontal_gaze = "RIGHT"
                        h_color = (0, 165, 255)  # Orange
                        eye_horizontal_margin = smoothed_horizontal - right_threshold
                    else:
                        eye_horizontal_gaze = "CENTER"
                        h_color = (255, 255, 0)
                        # Distance from nearest threshold
                        eye_horizontal_margin = min(abs(smoothed_horizontal - left_threshold), abs(smoothed_horizontal - right_threshold))

                    # Now also check face gaze position (smoothed)
                    face_vertical_gaze = "CENTER"
                    face_horizontal_gaze = "CENTER"
                    
                    # Define screen thirds for face gaze
                    third_h = self.SCREEN_HEIGHT // 3
                    two_thirds_h = 2 * third_h
                    third_w = self.SCREEN_WIDTH // 3
                    two_thirds_w = 2 * third_w
                    
                    # Determine face-based vertical gaze with margins
                    if face_gaze_position[1] < third_h:
                        face_vertical_gaze = "UP"
                        face_vertical_margin = third_h - face_gaze_position[1]
                    elif face_gaze_position[1] > two_thirds_h:
                        face_vertical_gaze = "DOWN"
                        face_vertical_margin = face_gaze_position[1] - two_thirds_h
                    else:
                        face_vertical_gaze = "CENTER"
                        # Distance from nearest boundary
                        face_vertical_margin = min(abs(face_gaze_position[1] - third_h), abs(face_gaze_position[1] - two_thirds_h))
                    
                    # Determine face-based horizontal gaze with margins
                    if face_gaze_position[0] < third_w:
                        face_horizontal_gaze = "LEFT"
                        face_horizontal_margin = third_w - face_gaze_position[0]
                    elif face_gaze_position[0] > two_thirds_w:
                        face_horizontal_gaze = "RIGHT"
                        face_horizontal_margin = face_gaze_position[0] - two_thirds_w
                    else:
                        face_horizontal_gaze = "CENTER"
                        # Distance from nearest boundary
                        face_horizontal_margin = min(abs(face_gaze_position[0] - third_w), abs(face_gaze_position[0] - two_thirds_w))
                    
                    # Combine iris and face gaze based on margins
                    # If both agree, use that direction
                    # If they disagree and have similar margins, use a hybrid segment
                    # If they disagree with different margins, use the one with bigger margin
                    
                    # Similarity threshold (if margins are within 20% of each other, consider them similar)
                    similarity_threshold = 0.5
                    
                    # Vertical gaze decision
                    if eye_vertical_gaze == face_vertical_gaze:
                        # Both agree
                        final_vertical_gaze = eye_vertical_gaze
                        use_hybrid_vertical = False
                    elif eye_vertical_gaze == "CENTER":
                        # Eye is center, use face
                        final_vertical_gaze = face_vertical_gaze
                        use_hybrid_vertical = False
                    elif face_vertical_gaze == "CENTER":
                        # Face is center, use eye
                        final_vertical_gaze = eye_vertical_gaze
                        use_hybrid_vertical = False
                    else:
                        # They disagree (e.g., one says UP, other says DOWN)
                        # Normalize margins to compare fairly
                        # For eye: normalize by range between calibration points
                        eye_v_range = calibration_height_up - calibration_height_down
                        normalized_eye_v_margin = eye_vertical_margin / eye_v_range if eye_v_range > 0 else 0
                        
                        # For face: normalize by screen height third
                        normalized_face_v_margin = face_vertical_margin / third_h if third_h > 0 else 0
                        
                        # Check if margins are similar
                        if normalized_eye_v_margin > 0 and normalized_face_v_margin > 0:
                            margin_ratio = min(normalized_eye_v_margin, normalized_face_v_margin) / max(normalized_eye_v_margin, normalized_face_v_margin)
                        else:
                            margin_ratio = 0
                        
                        if margin_ratio >= (1 - similarity_threshold):
                            # Margins are similar, use hybrid segment
                            final_vertical_gaze = "HYBRID"
                            use_hybrid_vertical = True
                            hybrid_vertical_combination = f"{eye_vertical_gaze}-{face_vertical_gaze}"
                        else:
                            # Use the one with bigger normalized margin
                            if normalized_eye_v_margin > normalized_face_v_margin:
                                final_vertical_gaze = eye_vertical_gaze
                            else:
                                final_vertical_gaze = face_vertical_gaze
                            use_hybrid_vertical = False
                    
                    # Horizontal gaze decision
                    if eye_horizontal_gaze == face_horizontal_gaze:
                        # Both agree
                        final_horizontal_gaze = eye_horizontal_gaze
                        use_hybrid_horizontal = False
                    elif eye_horizontal_gaze == "CENTER":
                        # Eye is center, use face
                        final_horizontal_gaze = face_horizontal_gaze
                        use_hybrid_horizontal = False
                    elif face_horizontal_gaze == "CENTER":
                        # Face is center, use eye
                        final_horizontal_gaze = eye_horizontal_gaze
                        use_hybrid_horizontal = False
                    else:
                        # They disagree (e.g., one says LEFT, other says RIGHT)
                        # Normalize margins to compare fairly
                        # For eye: normalize by range between calibration points
                        eye_h_range = calibration_horizontal_right - calibration_horizontal_left
                        normalized_eye_h_margin = eye_horizontal_margin / eye_h_range if eye_h_range > 0 else 0
                        
                        # For face: normalize by screen width third
                        normalized_face_h_margin = face_horizontal_margin / third_w if third_w > 0 else 0
                        
                        # Check if margins are similar
                        if normalized_eye_h_margin > 0 and normalized_face_h_margin > 0:
                            margin_ratio = min(normalized_eye_h_margin, normalized_face_h_margin) / max(normalized_eye_h_margin, normalized_face_h_margin)
                        else:
                            margin_ratio = 0
                        
                        if margin_ratio >= (1 - similarity_threshold):
                            # Margins are similar, use hybrid segment
                            final_horizontal_gaze = "HYBRID"
                            use_hybrid_horizontal = True
                            hybrid_horizontal_combination = f"{eye_horizontal_gaze}-{face_horizontal_gaze}"
                        else:
                            # Use the one with bigger normalized margin
                            if normalized_eye_h_margin > normalized_face_h_margin:
                                final_horizontal_gaze = eye_horizontal_gaze
                            else:
                                final_horizontal_gaze = face_horizontal_gaze
                            use_hybrid_horizontal = False

                    # Build combined gaze string
                    if use_hybrid_vertical and use_hybrid_horizontal:
                        # Both are hybrid - create extended region covering both possibilities
                        combined_gaze = f"HYBRID_{hybrid_vertical_combination}_{hybrid_horizontal_combination}"
                        combined_color = (255, 128, 0)  # Orange for hybrid
                    elif use_hybrid_vertical:
                        combined_gaze = f"HYBRID_{hybrid_vertical_combination} {final_horizontal_gaze}"
                        combined_color = (255, 128, 0)
                    elif use_hybrid_horizontal:
                        combined_gaze = f"{final_vertical_gaze} HYBRID_{hybrid_horizontal_combination}"
                        combined_color = (255, 128, 0)
                    elif final_vertical_gaze == "CENTER" and final_horizontal_gaze == "CENTER":
                        combined_gaze = "CENTER"
                        combined_color = (255, 255, 255)
                    else:
                        combined_gaze = f"{final_vertical_gaze} {final_horizontal_gaze}"
                        combined_color = (0, 255, 255)

            # ============================================ CALIBRATION & GAZE DETECTION =============================================== #

                # Screen quadrants visualization
                if calibration_stage == 6:
                    screen_frame = np.zeros((self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
                    
                    # For debugging: draw all segments
                    def draw_segment(rect_dims, color):
                        cv2.rectangle(screen_frame, (rect_dims[0], rect_dims[1]), 
                                      (rect_dims[2], rect_dims[3]), color, -1)

                    # Calculate dynamic thirds based on screen dimensions
                    spacing = 100  # Space between segments
                    third_w = self.SCREEN_WIDTH // 3
                    two_thirds_w = 2 * third_w
                    third_h = self.SCREEN_HEIGHT // 3
                    two_thirds_h = 2 * third_h
                    
                    # Determine which segment we're looking at based on combined_gaze
                    # Default to full screen if combined_gaze not set
                    if "HYBRID" in combined_gaze:
                        # Handle hybrid segments - use expanded regions with spacing
                        if "UP-DOWN" in combined_gaze or "DOWN-UP" in combined_gaze:
                            if "LEFT" in combined_gaze:
                                segment_rect_dimensions = (0, self.SCREEN_HEIGHT // 4, self.SCREEN_WIDTH // 2 - spacing // 2, 3 * self.SCREEN_HEIGHT // 4)
                            elif "RIGHT" in combined_gaze:
                                segment_rect_dimensions = (self.SCREEN_WIDTH // 2 + spacing // 2, self.SCREEN_HEIGHT // 4, self.SCREEN_WIDTH, 3 * self.SCREEN_HEIGHT // 4)
                            else:
                                segment_rect_dimensions = (third_w + spacing // 2, self.SCREEN_HEIGHT // 4, two_thirds_w - spacing // 2, 3 * self.SCREEN_HEIGHT // 4)
                        elif "LEFT-RIGHT" in combined_gaze or "RIGHT-LEFT" in combined_gaze:
                            if "UP" in combined_gaze:
                                segment_rect_dimensions = (self.SCREEN_WIDTH // 4, 0, 3 * self.SCREEN_WIDTH // 4, third_h - spacing // 2)
                            elif "DOWN" in combined_gaze:
                                segment_rect_dimensions = (self.SCREEN_WIDTH // 4, two_thirds_h + spacing // 2, 3 * self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT)
                            else:
                                segment_rect_dimensions = (self.SCREEN_WIDTH // 4, third_h + spacing // 2, 3 * self.SCREEN_WIDTH // 4, two_thirds_h - spacing // 2)
                        else:
                            segment_rect_dimensions = (self.SCREEN_WIDTH // 4, self.SCREEN_HEIGHT // 4, 3 * self.SCREEN_WIDTH // 4, 3 * self.SCREEN_HEIGHT // 4)

                    elif combined_gaze == "UP LEFT":
                        segment_rect_dimensions = (0, 0, self.SCREEN_WIDTH // 2 - spacing // 2, third_h - spacing // 2)
                    elif combined_gaze == "UP RIGHT":
                        segment_rect_dimensions = (self.SCREEN_WIDTH // 2 + spacing // 2, 0, self.SCREEN_WIDTH, third_h - spacing // 2)
                    elif combined_gaze == "DOWN LEFT":
                        segment_rect_dimensions = (0, two_thirds_h + spacing // 2, self.SCREEN_WIDTH // 2 - spacing // 2, self.SCREEN_HEIGHT)
                    elif combined_gaze == "DOWN RIGHT":
                        segment_rect_dimensions = (self.SCREEN_WIDTH // 2 + spacing // 2, two_thirds_h + spacing // 2, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
                    elif combined_gaze == "UP CENTER":
                        segment_rect_dimensions = (third_w + spacing // 2, 0, two_thirds_w - spacing // 2, third_h - spacing // 2)
                    elif combined_gaze == "DOWN CENTER":
                        segment_rect_dimensions = (third_w + spacing // 2, two_thirds_h + spacing // 2, two_thirds_w - spacing // 2, self.SCREEN_HEIGHT)
                    elif combined_gaze == "CENTER LEFT":
                        segment_rect_dimensions = (0, third_h + spacing // 2, self.SCREEN_WIDTH // 2 - spacing // 2, two_thirds_h - spacing // 2)
                    elif combined_gaze == "CENTER RIGHT":
                        segment_rect_dimensions = (self.SCREEN_WIDTH // 2 + spacing // 2, third_h + spacing // 2, self.SCREEN_WIDTH, two_thirds_h - spacing // 2)
                    elif combined_gaze == "CENTER":
                        segment_rect_dimensions = (third_w + spacing // 2, third_h + spacing // 2, two_thirds_w - spacing // 2, two_thirds_h - spacing // 2)
                    else:
                        # Default to full screen if gaze direction is unknown
                        segment_rect_dimensions = (0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

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
                    spacing = 100  # Space between segments
                    third_w = self.SCREEN_WIDTH // 3
                    two_thirds_w = 2 * third_w
                    third_h = self.SCREEN_HEIGHT // 3
                    two_thirds_h = 2 * third_h
                    
                    # Map normalized iris position to calibrated screen coordinates
                    # Horizontal mapping (left to right)
                    if avg_iris_x_norm < left_threshold:
                        # Looking left
                        h_range = left_threshold - calibration_horizontal_left
                        if h_range > 0:
                            norm_in_range = (avg_iris_x_norm - calibration_horizontal_left) / h_range
                            gaze_x = int(norm_in_range * third_w)  # Left third of screen
                        else:
                            gaze_x = third_w // 2
                    elif avg_iris_x_norm > right_threshold:
                        # Looking right
                        h_range = calibration_horizontal_right - right_threshold
                        if h_range > 0:
                            norm_in_range = (avg_iris_x_norm - right_threshold) / h_range
                            gaze_x = int(two_thirds_w + norm_in_range * third_w)  # Right third of screen
                        else:
                            gaze_x = two_thirds_w + third_w // 2
                    else:
                        # Looking center
                        h_range = right_threshold - left_threshold
                        if h_range > 0:
                            norm_in_range = (avg_iris_x_norm - left_threshold) / h_range
                            gaze_x = int(third_w + norm_in_range * third_w)  # Center third of screen
                        else:
                            gaze_x = self.SCREEN_WIDTH // 2

                    # Vertical mapping (up to down)
                    if avg_rect_height > up_threshold:
                        # Looking up
                        v_range = calibration_height_up - up_threshold
                        if v_range > 0:
                            norm_in_range = (avg_rect_height - up_threshold) / v_range
                            gaze_y = int(third_h - norm_in_range * third_h)  # Top third (inverted)
                        else:
                            gaze_y = third_h // 2
                    elif avg_rect_height < down_threshold:
                        # Looking down
                        v_range = down_threshold - calibration_height_down
                        if v_range > 0:
                            norm_in_range = (avg_rect_height - calibration_height_down) / v_range
                            gaze_y = int(two_thirds_h + norm_in_range * third_h)  # Bottom third
                        else:
                            gaze_y = two_thirds_h + third_h // 2
                    else:
                        # Looking center
                        v_range = up_threshold - down_threshold
                        if v_range > 0:
                            norm_in_range = (avg_rect_height - down_threshold) / v_range
                            gaze_y = int(third_h + norm_in_range * third_h)  # Center third
                        else:
                            gaze_y = self.SCREEN_HEIGHT // 2

                    # Clamp to screen bounds
                    gaze_x = max(0, min(self.SCREEN_WIDTH, gaze_x))
                    gaze_y = max(0, min(self.SCREEN_HEIGHT, gaze_y))

                    # Now map the gaze position to the specific segment
                    x1, y1, x2, y2 = segment_rect_dimensions
                    seg_w = x2 - x1
                    seg_h = y2 - y1

                    # Normalize gaze_x, gaze_y to [0,1] based on the full screen
                    norm_x = gaze_x / self.SCREEN_WIDTH
                    norm_y = gaze_y / self.SCREEN_HEIGHT

                    # Map normalized gaze to segment bounds
                    mapped_x = int(x1 + norm_x * seg_w)
                    mapped_y = int(y1 + norm_y * seg_h)

                    # Clamp to segment bounds
                    mapped_x = max(x1, min(x2 - 1, mapped_x))
                    mapped_y = max(y1, min(y2 - 1, mapped_y))
                    
                    # Update gaze position
                    iris_gaze_position = (mapped_x, mapped_y)
                    iris_gaze_position_history.append(iris_gaze_position)

                    if len(iris_gaze_position_history) > 0:
                        # Smooth gaze position
                        smoothed_x = int(np.mean([pos[0] for pos in iris_gaze_position_history]))
                        smoothed_y = int(np.mean([pos[1] for pos in iris_gaze_position_history]))
                        iris_gaze_position = (smoothed_x, smoothed_y)
                
                # ============================================ GAZE POSITION DISPLAY =============================================== #

                # Gaze position display
                if calibration_stage == 6:

                    face_weight = 0.40 # 40% face-based tracking
                    eye_weight = 0.60 # 60% eye-based tracking
                    
                    weighted_gaze_x = int((face_gaze_position[0] * face_weight + iris_gaze_position[0] * eye_weight))
                    weighted_gaze_y = int((face_gaze_position[1] * face_weight + iris_gaze_position[1] * eye_weight))

                    # With this (normalized version):
                    weighted_gaze_x_norm = weighted_gaze_x / self.SCREEN_WIDTH
                    weighted_gaze_y_norm = weighted_gaze_y / self.SCREEN_HEIGHT
                    self.weighted_screen_position = (weighted_gaze_x_norm, weighted_gaze_y_norm)

            # ============================================ FINAL VARIABLE UPDATES =============================================== #
                
                self.calibration_offset_yaw = calibration_offset_yaw
                self.calibration_offset_pitch = calibration_offset_pitch
                self.calibration_height_up = calibration_height_up
                self.calibration_height_center = calibration_height_center
                self.calibration_height_down = calibration_height_down
                self.calibration_horizontal_left = calibration_horizontal_left
                self.calibration_horizontal_center = calibration_horizontal_center
                self.calibration_horizontal_right = calibration_horizontal_right
                self.is_up_calibrated = is_up_calibrated
                self.is_center_calibrated = is_center_calibrated
                self.is_down_calibrated = is_down_calibrated
                self.is_left_calibrated = is_left_calibrated
                self.is_h_center_calibrated = is_h_center_calibrated
                self.is_right_calibrated = is_right_calibrated
                self.up_threshold = up_threshold
                self.down_threshold = down_threshold    
                self.left_threshold = left_threshold
                self.right_threshold = right_threshold
                self.distance_threshold = distance_threshold
                self.combined_gaze = combined_gaze
                self.iris_gaze_position = iris_gaze_position

            # ============================================ FINAL VARIABLE UPDATES =============================================== #
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()

    def get_screen_position(self):
        return self.weighted_screen_position
       

if __name__ == "__main__":
    camera_track = CameraTrack()

    # Load existing calibration from json
    try:
        with open("calibration_data.json", "r") as f:
            calibration_data = json.load(f)
            camera_track.load_calibration(calibration_data)
    except FileNotFoundError:
        print("No calibration data found. Starting fresh.")

    try:
        while True:
            frame = camera_track.get_frame()
            # Convert byte data back to image for display
            nparr = np.frombuffer(frame, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            cv2.imshow('Camera Feed', frame)
            
            screen_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            screen_pos = camera_track.get_screen_position()
            gaze_x = int(screen_pos[0] * 1920)
            gaze_y = int(screen_pos[1] * 1080)
            # 80px circle outline for weighted gaze 
            cv2.circle(screen_frame, (gaze_x, gaze_y), 100, (0, 255, 255), 3)
                
            # Peripheral circles for reference
            cv2.circle(screen_frame, (gaze_x, gaze_y), 200, (255, 255, 255), 2)

            cv2.imshow('Weighted Gaze Position', screen_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                camera_track.update_calibration_stage(camera_track.calibration_stage + 1)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                camera_track.save_calibration()
            if cv2.waitKey(1) & 0xFF == ord('l'):
                camera_track.load_calibration()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera_track.release()