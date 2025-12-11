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
        camera_instance = camera_feed()
        # Load calibration immediately if available
        try:
            from django.contrib.auth.models import AnonymousUser
            # We can't get request.user here, so check in calling function
        except:
            pass
    return camera_instance

def is_camera_initialized():
    """Helper function to check if camera instance exists"""
    global camera_instance
    return camera_instance is not None

@login_required_role(allowed_roles=['Student', 'Admin'])
def reset_calibration(request):
    global camera_instance
    if not is_camera_initialized():
        return JsonResponse({
            'status': 'error',
            'message': 'Camera not initialized',
            'isInitialized': False
        })
    
    camera_instance.reset_calibration()
    return JsonResponse({
        'status': 'Calibration reset successfully.',
        'isInitialized': True
    })

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
    if not is_camera_initialized():
        return JsonResponse({
            'mid_x': 0,
            'mid_y': 0,
            'isInitialized': False
        })
    
    cam = get_camera()
    gaze_x, gaze_y = cam.get_screen_position()
    return JsonResponse({
        'mid_x': gaze_x,
        'mid_y': gaze_y,
        'isInitialized': True
    })
    
def gen(cam_capture):
    while True:
        frame = cam_capture.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + 
              b'\r\n\r\n')

@login_required_role(allowed_roles=['Student', 'Admin'])
def video_feed(request):
    # Ensure camera is initialized before streaming
    if not is_camera_initialized():
        print("WARNING: video_feed called before camera initialized")
        return JsonResponse({'error': 'Camera not initialized'}, status=503)
    
    return StreamingHttpResponse(gen(get_camera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')

@login_required_role(allowed_roles=['Student', 'Admin'])
def update_calibration(request):
    global next_point
    global camera_instance
    
    if not is_camera_initialized():
        camera_instance = get_camera()
    
    # Move the logic outside the if-else
    new_stage = camera_instance.calibration_stage + 1
    if new_stage > 6:
        new_stage = 6  # Max stage is 6 (tracking mode)
    camera_instance.update_calibration_stage(new_stage)
    next_point = get_next_point_text(camera_instance.calibration_stage)
    
    return JsonResponse({
        'next_point': next_point, 
        'calibration_stage': camera_instance.calibration_stage,
        'isInitialized': True
    })


@login_required_role(allowed_roles=['Student', 'Admin'])
def release_camera(request):
    global camera_instance
    if not is_camera_initialized():
        return JsonResponse({
            'status': 'Camera already released or not initialized.',
            'isInitialized': False
        })
    
    try:
        camera_instance.release()
        camera_instance = None
        return JsonResponse({
            'status': 'Camera released successfully.',
            'isInitialized': False
        })
    except Exception as e:
        print(f"Error releasing camera: {e}")
        return JsonResponse({
            'status': f'Error releasing camera: {str(e)}',
            'isInitialized': True
        })

@login_required_role(allowed_roles=['Student', 'Admin'])
def save_calibration(request):
    global camera_instance
    if not is_camera_initialized():
        return JsonResponse({
            'status': 'error',
            'message': 'Camera not initialized',
            'isInitialized': False
        })
    
    # Save calibration data algorithm
    calibration_data = camera_instance.save_calibration()
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    student_profile.camera_calibration_data = calibration_data
    student_profile.save()

    return JsonResponse({
        'status': 'Calibration data saved successfully.',
        'calibration_data': calibration_data,
        'isInitialized': True
    })



@login_required_role(allowed_roles=['Student', 'Admin'])
def load_calibration(request):
    global camera_instance
    if not is_camera_initialized():
        camera_instance = get_camera()
    
    # Load calibration data algorithm
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    calibration_data = student_profile.camera_calibration_data
    camera_instance.load_calibration(calibration_data)
    
    return JsonResponse({
        'status': 'Calibration data loaded successfully.',
        'calibration_data': calibration_data,
        'isInitialized': True
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def update_calibration_status(request):
    global camera_instance
    if not is_camera_initialized():
        return JsonResponse({
            'status': False,
            'isInitialized': False,
            'message': 'Camera not initialized'
        })
    
    stage = camera_instance.calibration_stage
    
    # Check if samples have been collected (60 samples = calibrated)
    if stage == -1:
        is_point_calibrated = True
    elif stage == 0:
        is_point_calibrated = len(camera_instance.calibration_up_samples) >= 60
    elif stage == 1:
        is_point_calibrated = len(camera_instance.calibration_center_samples) >= 60
    elif stage == 2:
        is_point_calibrated = len(camera_instance.calibration_down_samples) >= 60
    elif stage == 3:
        is_point_calibrated = len(camera_instance.calibration_left_samples) >= 60
    elif stage == 4:
        is_point_calibrated = len(camera_instance.calibration_h_center_samples) >= 60
    elif stage == 5:
        is_point_calibrated = len(camera_instance.calibration_right_samples) >= 60
    elif stage >= 6:
        is_point_calibrated = True
    else:
        is_point_calibrated = False
    
    return JsonResponse({
        'status': is_point_calibrated,
        'isInitialized': True
    })
    
# get calibration stage
@login_required_role(allowed_roles=['Student', 'Admin'])
def get_calibration_stage(request):
    global camera_instance
    
    # Always release old instance on page load to prevent resource leaks
    if is_camera_initialized():
        try:
            camera_instance.release()
            print("Released existing camera instance")
        except Exception as e:
            print(f"Error releasing camera: {e}")
        camera_instance = None
    
    # Create fresh instance
    camera_instance = get_camera()
    print("Created new camera instance")
    
    # Give camera time to initialize
    import time
    time.sleep(0.5)
    
    # Load saved calibration
    try:
        student_profile = StudentProfile.objects.get(profile__user=request.user)
        calibration_data = student_profile.camera_calibration_data
        
        if calibration_data and isinstance(calibration_data, dict) and calibration_data:
            # Verify calibration data has all required fields
            required_fields = [
                'calibration_height_up', 'calibration_height_center', 'calibration_height_down',
                'calibration_horizontal_left', 'calibration_horizontal_center', 'calibration_horizontal_right',
                'calibration_offset_yaw', 'calibration_offset_pitch',
                'up_threshold', 'down_threshold', 'left_threshold', 'right_threshold',
                'distance_threshold'
            ]
            
            if all(field in calibration_data for field in required_fields):
                camera_instance.load_calibration(calibration_data)
                print(f"Loaded calibration for {student_profile.profile.username}")
                return JsonResponse({
                    'calibration_stage': camera_instance.calibration_stage,
                    'isInitialized': True,
                    'calibrationLoaded': True
                })
            else:
                print("Calibration data incomplete, resetting")
                camera_instance.reset_calibration()
        else:
            print("No valid calibration data found, resetting")
            camera_instance.reset_calibration()
            
    except StudentProfile.DoesNotExist:
        print(f"Student profile not found for user {request.user.username}")
        camera_instance.reset_calibration()
    except Exception as e:
        print(f"Error loading calibration: {e}")
        camera_instance.reset_calibration()
    
    return JsonResponse({
        'calibration_stage': camera_instance.calibration_stage,
        'isInitialized': True,
        'calibrationLoaded': False
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
    if is_camera_initialized():
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
                'calibration_stage': camera_instance.calibration_stage,
                'isInitialized': True
            })
        else:
            return JsonResponse({
                'status': 'warning',
                'message': 'Camera reloaded but no calibration data found. Please calibrate first.',
                'calibration_stage': camera_instance.calibration_stage,
                'isInitialized': True
            })
    except StudentProfile.DoesNotExist:
        return JsonResponse({
            'status': 'error',
            'message': 'Student profile not found.',
            'isInitialized': False
        }, status=404)
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error loading calibration: {str(e)}',
            'isInitialized': True
        }, status=500)
