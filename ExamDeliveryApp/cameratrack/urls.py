from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns for the cameratrack app here
    path('', views.camera_track, name='camera-track'),
    path('api/video_feed/', views.video_feed, name='video'),
    path('api/get_screen_position/', views.get_screen_position, name='get-screen-position'),
    path('api/update_calibration/', views.update_calibration, name='update-calibration'),
    path('api/update_calibration_status/', views.update_calibration_status, name='update-calibration-status'),
    path('api/save_calibration/', views.save_calibration, name='save-calibration'),
    path('api/release_camera/', views.release_camera, name='release-camera'),
    path('api/reset_calibration/', views.reset_calibration, name='reset-calibration'),
    path('api/get_calibration_stage/', views.get_calibration_stage, name='get-calibration-stage'),
    path('api/send_active_status/', views.send_active_status, name='send-active-status'),
    path('api/reload_camera_with_calibration/', views.reload_camera_with_calibration, name='reload-camera-with-calibration'),
]