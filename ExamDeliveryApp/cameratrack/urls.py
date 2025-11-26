from django.urls import path
from . import views

urlpatterns = [
    # Define your URL patterns for the cameratrack app here
    path('track/', views.camera_track, name='camera-track'),
    path('video_feed/', views.video_feed, name='video'),
    path('eye_midpoint/', views.eye_midpoint, name='eye-midpoint'),
]