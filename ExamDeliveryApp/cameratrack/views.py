from django.shortcuts import render
from cameratrack import camera_feed
from django.http import StreamingHttpResponse
from django.http import JsonResponse

camera_instance = None

# Create your views here.
def camera_track(request):
    return render(request, 'cameratrack/cameratrack.html')

def get_camera():
    global camera_instance
    if camera_instance is None:
        camera_instance = camera_feed.Video()
    return camera_instance

def eye_midpoint(request):
    cam = get_camera()
    mid_x, mid_y = cam.get_eye_midpoint()

    return JsonResponse({'mid_x': mid_x, 'mid_y': mid_y})
    

def gen(cam_capture):
    while True:
        frame = cam_capture.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + 
              b'\r\n\r\n')
        
def video_feed(request):
    return StreamingHttpResponse(gen(get_camera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')