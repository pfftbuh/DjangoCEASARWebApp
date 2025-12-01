from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns for the cameratrack app here
    path('', views.camera_track, name='camera-track'),
    path('api/video_feed/', views.video_feed, name='video'),
    path('api/get_screen_position/', views.get_screen_position, name='get-screen-position'),
    path('api/update_calibration/', views.update_calibration, name='update-calibration'),
    path('api/update_calibration_status/', views.update_calibration_status, name='update-calibration-status'),
]