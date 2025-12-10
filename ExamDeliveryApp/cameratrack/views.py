from django.shortcuts import render
from .camera_track import CameraTrack as camera_feed
from django.http import StreamingHttpResponse
from django.http import JsonResponse
from home.models import Profile, StudentProfile, ActiveExamSessions
from home.decorators import login_required_role


camera_instance = None
next_point = None


# Create your views here.
@login_required_role(allowed_roles=['Student', 'Admin'])
def camera_track(request):
    return render(request, 'cameratrack/cameratrack.html')

def get_camera():
    global camera_instance
    if camera_instance is None:
        try:
            camera_instance = camera_feed()
            # Test if camera is working
            test_frame = camera_instance.get_frame()
            if test_frame is None:
                raise ValueError("Camera not accessible")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Wait a bit and try again
            import time
            time.sleep(0.5)
            try:
                camera_instance = camera_feed()
            except:
                camera_instance = None
                raise
    return camera_instance


@login_required_role(allowed_roles=['Student', 'Admin'])
def reset_calibration(request):
    global camera_instance
    if camera_instance is not None:
        camera_instance.reset_calibration()
    else:
        camera_instance = get_camera()
        camera_instance.reset_calibration()

    return JsonResponse({'status': 'Calibration reset successfully.'})

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

@login_required_role(allowed_roles=['Student', 'Admin'])
def get_screen_position(request):
    cam = get_camera()
    gaze_x, gaze_y = cam.get_screen_position()
    return JsonResponse({'mid_x': gaze_x, 'mid_y': gaze_y})
    
def gen(cam_capture):
    while True:
        frame = cam_capture.get_frame()
        # You can include face_distance in the frame if needed, e.g., overlay text
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + 
              b'\r\n\r\n')

@login_required_role(allowed_roles=['Student', 'Admin'])
def video_feed(request):
    return StreamingHttpResponse(gen(get_camera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

@login_required_role(allowed_roles=['Student', 'Admin'])
def update_calibration(request):
    global next_point
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    
    # Get current stage
    current_stage = camera_instance.calibration_stage
    
    # Clear the samples for the CURRENT stage before moving to next
    if current_stage == -1:
        camera_instance.calibration_up_samples.clear()
    elif current_stage == 0:
        camera_instance.calibration_center_samples.clear()
    elif current_stage == 1:
        camera_instance.calibration_down_samples.clear()
    elif current_stage == 2:
        camera_instance.calibration_left_samples.clear()
    elif current_stage == 3:
        camera_instance.calibration_h_center_samples.clear()
    elif current_stage == 4:
        camera_instance.calibration_right_samples.clear()
    
    # Move to next stage
    new_stage = current_stage + 1
    if new_stage > 6:
        new_stage = 6  # Max stage is 6 (tracking mode)
    
    # Update the stage
    camera_instance.calibration_stage = new_stage  # Direct assignment instead of method
    next_point = get_next_point_text(new_stage)
    
    print(f"DEBUG: Updated calibration stage from {current_stage} to {new_stage}")  # Debug log
    
    return JsonResponse({
        'next_point': next_point, 
        'calibration_stage': new_stage
    })


# Remove the decorator
def release_camera(request):
    global camera_instance
    if camera_instance is not None:
        try:
            camera_instance.release()
            camera_instance = None
        except Exception as e:
            print(f"Error releasing camera: {e}")

    return JsonResponse({
        'status': 'Camera released successfully.'
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def save_calibration(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    
    # Save calibration data algorithm
    calibration_data = camera_instance.save_calibration()
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    # Fixed: use camera_calibration_data instead of calibration_data
    student_profile.camera_calibration_data = calibration_data
    student_profile.save()

    return JsonResponse({
        'status': 'Calibration data saved successfully.',
        'calibration_data': calibration_data
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def load_calibration(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    
    # Load calibration data algorithm
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    calibration_data = student_profile.camera_calibration_data
    camera_instance.load_calibration(calibration_data)
    
    return JsonResponse({
        'status': 'Calibration data loaded successfully.',
        'calibration_data': calibration_data
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def get_face_distance(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
    
    face_distance_cm = camera_instance.get_face_distance()
    
    return JsonResponse({
        'face_distance_cm': face_distance_cm
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def update_calibration_status(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
        is_point_calibrated = False
    else:
        stage = camera_instance.calibration_stage
        
        # Check if samples have been collected (60 samples = calibrated)
        if stage == -1:
            # Stage -1 is preparation - user can proceed immediately when ready
            is_point_calibrated = True
        elif stage == 0:
            # At stage 0, check if "up" samples are collected
            is_point_calibrated = len(camera_instance.calibration_up_samples) >= 60
        elif stage == 1:
            # At stage 1, check if "center" samples are collected
            is_point_calibrated = len(camera_instance.calibration_center_samples) >= 60
        elif stage == 2:
            # At stage 2, check if "down" samples are collected
            is_point_calibrated = len(camera_instance.calibration_down_samples) >= 60
        elif stage == 3:
            # At stage 3, check if "left" samples are collected
            is_point_calibrated = len(camera_instance.calibration_left_samples) >= 60
        elif stage == 4:
            # At stage 4, check if "h_center" samples are collected
            is_point_calibrated = len(camera_instance.calibration_h_center_samples) >= 60
        elif stage == 5:
            # At stage 5, check if "right" samples are collected
            is_point_calibrated = len(camera_instance.calibration_right_samples) >= 60
        elif stage >= 6:
            # Tracking mode
            is_point_calibrated = True
        else:
            is_point_calibrated = False
    
    return JsonResponse({
        'status': is_point_calibrated
    })
    
# get calibration stage
@login_required_role(allowed_roles=['Student', 'Admin'])
def get_calibration_stage(request):
    global camera_instance
    if camera_instance is None:
        camera_instance = get_camera()
        camera_instance.reset_calibration()
    
    return JsonResponse({
        'calibration_stage': camera_instance.calibration_stage
    })

# send active status
@login_required_role(allowed_roles=['Student', 'Admin'])
def send_active_status(request):
    user_profile = Profile.objects.get(user=request.user)
    try:
        active_session = ActiveExamSessions.objects.get(student_profile__profile=user_profile)
        is_active = active_session.is_active
    except ActiveExamSessions.DoesNotExist:
        is_active = False

    return JsonResponse({
        'is_active': is_active
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def reload_camera_with_calibration(request):
    """Release current camera, create new instance, and load student's calibration data"""
    global camera_instance
    
    # Release existing camera if any
    if camera_instance is not None:
        camera_instance.release()
        camera_instance = None
    
    # Create new camera instance
    camera_instance = get_camera()
    
    # Load calibration data from student profile
    try:
        student_profile = StudentProfile.objects.get(profile__user=request.user)
        calibration_data = student_profile.camera_calibration_data
        
        if calibration_data and isinstance(calibration_data, dict) and calibration_data:
            camera_instance.load_calibration(calibration_data)
            return JsonResponse({
                'status': 'success',
                'message': 'Camera reloaded and calibration data loaded successfully.',
                'calibration_stage': camera_instance.calibration_stage
            })
        else:
            return JsonResponse({
                'status': 'warning',
                'message': 'Camera reloaded but no calibration data found. Please calibrate first.',
                'calibration_stage': camera_instance.calibration_stage
            })
    except StudentProfile.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Student profile not found.'
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error loading calibration: {str(e)}'
        }, status=500)
