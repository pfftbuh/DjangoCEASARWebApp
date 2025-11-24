from django.urls import path
from . import views

urlpatterns = [
    path('', view=views.teacher_dashboard, name='teacher-home'),
    path('profile/', view=views.teacher_profile, name='teacher-profile'),
    path('exams/manage/', view=views.teacher_manage_exams, name='teacher-manage-exams'),
    path('exams/create/', view=views.teacher_create_exam, name='teacher-create-exam'),
    path('exams/submissions/<int:exam_id>/', view=views.teacher_view_submissions, name='teacher-view-submissions'),
    path('exams/grade/<int:submission_id>/', view=views.teacher_grade_submission, name='teacher-grade-submission'),
    path('reports/generate/', view=views.teacher_generate_reports, name='teacher-generate-reports'),
    path('help/', view=views.teacher_help, name='teacher-help'),
    path('exams/edit/<int:exam_id>/', view=views.teacher_modify_exam, name='teacher-modify-exam'),
    path('exams/start/<int:exam_id>/', view=views.teacher_create_exam, name='teacher-start-exam'),
    path('exam/<int:exam_id>/delete-question/<int:question_id>/', views.teacher_delete_question, name='teacher-delete-question'),
    path('exams/edit/details/<int:exam_id>/', view=views.teacher_modify_exam_details, name='teacher-modify-exam-details'),
    path('violation/logs/', view=views.teacher_violation_logs, name='teacher-violation-logs'),
    
    ## FOR TESTING PURPOSES ONLY
    path('violation/check/', view=views.violation_check, name='violation-check'),
    path('exam/start/debug/', view=views.teacher_start_exam, name='teacher-start-exam'),
    path('exam/start/debug/submit', view=views.teacher_start_exam, name='teacher-submit-exam'),
]