from eyetrax import GazeEstimator, run_9_point_calibration
import cv2
import numpy as np

# Get screen dimensions
import tkinter as tk
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Create estimator and calibrate
estimator = GazeEstimator()
run_9_point_calibration(estimator)

# Save model
estimator.save_model("gaze_model.pkl")

# Load model
estimator = GazeEstimator()
estimator.load_model("gaze_model.pkl")

cap = cv2.VideoCapture(0)

# Create visualization window
viz_width = 1920
viz_height = 1080
viz_frame = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)

while True:
    # Extract features from frame
    ret, frame = cap.read()
    features, blink = estimator.extract_features(frame)

    # Predict screen coordinates
    if features is not None and not blink:
        x, y = estimator.predict([features])[0]
        print(f"Gaze: ({x:.0f}, {y:.0f})")
        
        # Map gaze coordinates to visualization frame
        mapped_x = int((x / screen_width) * viz_width)
        mapped_y = int((y / screen_height) * viz_height)
        
        # Clamp values to window bounds
        mapped_x = max(0, min(viz_width - 1, mapped_x))
        mapped_y = max(0, min(viz_height - 1, mapped_y))
        
        # Clear and redraw visualization
        viz_frame = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
        cv2.circle(viz_frame, (mapped_x, mapped_y), 15, (0, 255, 0), -1)
        cv2.putText(viz_frame, f"Gaze: ({x:.0f}, {y:.0f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display visualization
    cv2.imshow("Gaze Tracking", viz_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()