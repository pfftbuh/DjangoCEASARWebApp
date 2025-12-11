from django.urls import path
from studentside import views
from cameratrack import views as cam_views



urlpatterns = [
    path('', view=views.student_dashboard, name='student-home'),
    path('profile/', view=views.student_profile, name='student-profile'),
    path('exams/', view=views.student_exams_tab, name='student-exams'),
    path('exams/take/<int:exam_id>/', view=views.student_take_exam, name='student-take-exam'),
    path('exams/calibrate/<int:exam_id>/', view=views.student_cam_calibration, name='student-cam-calibration'),
    path('exams/results/<int:exam_id>/', view=views.student_view_results, name='student-view-results'),
    path('help/', view=views.student_help, name='student-help'),
    path('submit_exam/', view=views.student_submit_exam, name='student-submit-exam'),
    path('grades/', view=views.student_grades_view, name='student-grades'),
    path('exams/details/<int:exam_id>/', view=views.student_exam_details, name='student-exam-details'),
    path('violation_check/<int:exam_id>/', view=views.violation_check, name='violation-check'),
    path('exams/calibrate/<int:exam_id>/', view=views.student_cam_calibration, name='student-exam-calibration'),

    path('api/camera', cam_views.camera_track, name='camera-track'),
    path('api/video_feed/', cam_views.video_feed, name='video'),
    path('api/get_screen_position/', cam_views.get_screen_position, name='get-screen-position'),
    path('api/update_calibration/', cam_views.update_calibration, name='update-calibration'),
    path('api/update_calibration_status/', cam_views.update_calibration_status, name='update-calibration-status'),
    path('api/save_calibration/', cam_views.save_calibration, name='save-calibration'),
    path('api/release_camera/', cam_views.release_camera, name='release-camera'),
    path('api/load_calibration/', cam_views.load_calibration, name='load-calibration-data'),
    path('api/reset_calibration/', cam_views.reset_calibration, name='reset-calibration'),
    path('api/get_calibration_stage/', cam_views.get_calibration_stage, name='get-calibration-stage'),
    path('api/send_active_status/', cam_views.send_active_status, name='send-active-status'),
    path('api/reload_camera_with_calibration/', cam_views.reload_camera_with_calibration, name='reload-camera-with-calibration'),
]
    