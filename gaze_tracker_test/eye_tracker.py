import cv2
import numpy as np
import mediapipe as mp

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

# Cropped Eye Frame Padding
padding = 5




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
            for idx in range(473, 478):  # Iris landmarks for the right eye
                landmark = landmarks[idx]
                x = landmark.x * w
                y = landmark.y * h
                right_eye_points.append((x, y))
                # cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            
            # Left eye: landmarks 469-472
            for idx in range(468, 473):  # Iris landmarks for the left eye
                landmark = landmarks[idx]
                x = landmark.x * w
                y = landmark.y * h
                left_eye_points.append((x, y))
                # cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
                

            for idx in left_eye_indices + right_eye_indices:
                landmark = landmarks[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                all_eye_points.append((x, y))
                all_eye_idx.append(idx)
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)
                
                # Draw index as text in frame
                #cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            

            # Calculate bounding box around both eyes
            all_eye_points = np.array(all_eye_points)
            x_min = max(0, int(all_eye_points[:, 0].min()) - padding - 20)
            x_max = min(w, int(all_eye_points[:, 0].max()) + padding + 20)
            y_min = max(0, int(all_eye_points[:, 1].min()) - padding - 10)
            y_max = min(h, int(all_eye_points[:, 1].max()) + padding + 10)


            # Crop the eye region from the frame
            eye_frame = frame[y_min:y_max, x_min:x_max].copy()

            # Convert right eye points to eye_frame coordinates
            right_eye_points = np.array(right_eye_points) - np.array([x_min, y_min])
            left_eye_points = np.array(left_eye_points) - np.array([x_min, y_min])

            
            # Create a rectangle with landmarks 160, 153 for left eye
            # left eye rectangle
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

            # Mid point x
            r_mid_x = (re_x1 + re_x2) // 2
            r_mid_y = (re_y1 + re_y2) // 2

            # Convert to eye_frame coordinates
            r_mid_x -= x_min
            r_mid_y -= y_min
            ef_re_y2 = re_y2 - y_min
            ef_re_y1 = re_y1 - y_min
            ef_re_x1 = re_x1 - x_min
            ef_re_x2 = re_x2 - x_min

            #cross in the middle of rectangle
            cv2.line(eye_frame, (r_mid_x, ef_re_y1), (r_mid_x, ef_re_y2), (255, 0 , 0), 1)
            cv2.line(eye_frame, (ef_re_x1, r_mid_y), (ef_re_x2, r_mid_y), (255, 0 , 0), 1)

            # Draw all eye landmarks on the cropped eye frame
            le_points = np.array(le_points_list) - np.array([x_min, y_min])
            re_points = np.array(re_points_list) - np.array([x_min, y_min])
            
            cv2.rectangle(eye_frame, tuple(le_points[0]), tuple(le_points[1]), (255, 0, 0), 1)
            cv2.rectangle(eye_frame, tuple(re_points[0]), tuple(re_points[1]), (255, 0, 0), 1)

            # all eye points in eye_frame coordinates
            all_eye_points = np.array(all_eye_points) - np.array([x_min, y_min])
            for point in all_eye_points:
                cv2.circle(eye_frame, tuple(point), 1, (0, 255, 255), -1)
                # write index number
                idx = all_eye_idx[all_eye_points.tolist().index(point.tolist())]
                if idx in (160, 153, 385, 373):
                    cv2.putText(eye_frame, str(idx), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Draw center dot for right eye iris
            right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(right_eye_center), 2, (0, 0, 255), 2)

            # Draw center dot for left eye iris
            left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
            cv2.circle(eye_frame, tuple(left_eye_center), 2, (0, 0, 255), 2)
                

            eye_frame = cv2.resize(eye_frame, (635, 240))          
            cv2.imshow('Cropped Eyes', eye_frame)

            
            center = np.mean(right_eye_points + left_eye_points, axis=0).astype(int)
            cv2.circle(frame, tuple(center), 3, (255, 0, 0), -1)

            
    

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Eye Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break