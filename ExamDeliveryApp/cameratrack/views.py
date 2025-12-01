from django.shortcuts import render
from .camera_track import CameraTrack as camera_feed
from django.http import StreamingHttpResponse
from django.http import JsonResponse

camera_instance = None
next_point = None

# Create your views here.
def camera_track(request):
    return render(request, 'cameratrack/cameratrack.html')

def get_camera():
    global camera_instance
    if camera_instance is None:
        camera_instance = camera_feed()
    return camera_instance

def get_next_point_text(stage):
    """Helper function to get the next calibration point text"""
    if stage == -1:
        return "Center Up"
    elif stage == 0:
        return "Center"
    elif stage == 1:
        return "Center Down"
    elif stage == 2:
        return "Left Center"
    elif stage == 3:
        return "Center"
    elif stage == 4:
        return "Right Center"
    elif stage == 5:
        return "Calibration Complete"
    else:
        return "Start Tracking"

def get_screen_position(request):
    cam = get_camera()
    gaze_x, gaze_y = cam.get_screen_position()
    return JsonResponse({'mid_x': gaze_x, 'mid_y': gaze_y})
    
def gen(cam_capture):
    while True:
        frame = cam_capture.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + 
              b'\r\n\r\n')
        
def video_feed(request):
    return StreamingHttpResponse(gen(get_camera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

def update_calibration(request):
    global next_point
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    else:
        is_calibrated = False
        new_stage = camera_instance.calibration_stage + 1
        if new_stage > 6:
            new_stage = 6  # Max stage is 6 (tracking mode)
        camera_instance.update_calibration_stage(new_stage)
        next_point = get_next_point_text(camera_instance.calibration_stage)
        
        return JsonResponse({
            'next_point': next_point, 
            'calibration_stage': camera_instance.calibration_stage
        })


def save_calibration(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    else:
        # Save calibration data algorithm
        camera_instance.save_calibration_data()
        return JsonResponse({
            'status': 'Calibration data saved successfully.'
        })

def load_calibration(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    else:
        # Load calibration data algorithm

        return JsonResponse({
            'status': 'Calibration data loaded successfully.'
        })

def update_calibration_status(request):

    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
        is_point_calibrated = False
    else:
        if camera_instance.calibration_stage == -1:
            is_point_calibrated = camera_instance.is_up_calibrated
        elif camera_instance.calibration_stage == 0:
            is_point_calibrated = camera_instance.is_up_calibrated
        elif camera_instance.calibration_stage == 1:
            is_point_calibrated = camera_instance.is_center_calibrated
        elif camera_instance.calibration_stage == 2:
            is_point_calibrated = camera_instance.is_down_calibrated
        elif camera_instance.calibration_stage == 3:
            is_point_calibrated = camera_instance.is_left_calibrated
        elif camera_instance.calibration_stage == 4:
            is_point_calibrated = camera_instance.is_h_center_calibrated
        elif camera_instance.calibration_stage == 5:
            is_point_calibrated = camera_instance.is_right_calibrated
        elif camera_instance.calibration_stage >= 6:
            is_point_calibrated = True  # Tracking mode, considered calibrated
        return JsonResponse({
            'status': is_point_calibrated
        })


   