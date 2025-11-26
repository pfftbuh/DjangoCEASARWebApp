import cv2
import mediapipe as mp
import numpy as np

class Video(object):
    face_mesh=mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __def__(self):
        self.video.release()
    def get_frame(self):
        ret, frame=self.video.read()
        frame=cv2.flip(frame,1)
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        output=self.face_mesh.process(rgb_frame)
        landmark_points=output.multi_face_landmarks
        frame_h , frame_w, _ = frame.shape
        if landmark_points:
            landmarks = landmark_points[0].landmark
            points = []
            # Track both eyes: right eye (landmarks 474-477), left eye (landmarks 469-472)
            left_eye_points = []
            right_eye_points = []
            
            # Right eye: landmarks 474-477
            for idx in range(474, 478):
                landmark = landmarks[idx]
                x = landmark.x * frame_w
                y = landmark.y * frame_h
                right_eye_points.append((x, y))
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            
            # Left eye: landmarks 469-472
            for idx in range(469, 473):
                landmark = landmarks[idx]
                x = landmark.x * frame_w
                y = landmark.y * frame_h
                left_eye_points.append((x, y))
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

            # Calculate center of each eye
            if left_eye_points and right_eye_points:
                left_eye_center = (
                    int(sum([p[0] for p in left_eye_points]) / len(left_eye_points)),
                    int(sum([p[1] for p in left_eye_points]) / len(left_eye_points))
                )
                right_eye_center = (
                    int(sum([p[0] for p in right_eye_points]) / len(right_eye_points)),
                    int(sum([p[1] for p in right_eye_points]) / len(right_eye_points))
                )
                # Draw circles at each eye center
                cv2.circle(frame, left_eye_center, 2, (255, 0, 0), -1)
                cv2.circle(frame, right_eye_center, 2, (255, 0, 0), -1)

                # Draw a circle at the middle point between both eyes
                mid_x = int((left_eye_center[0] + right_eye_center[0]) / 2)
                mid_y = int((left_eye_center[1] + right_eye_center[1]) / 2)
                cv2.circle(frame, (mid_x, mid_y), 3, (0, 0, 255), -1)

                self.mid_x = mid_x
                self.mid_y = mid_y

        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
    
    def get_eye_midpoint(self):
        return (self.mid_x, self.mid_y)