from django.urls import path
from . import views

urlpatterns = [
    path('', view=views.student_dashboard, name='student-home'),
    path('profile/', view=views.student_profile, name='student-profile'),
    path('exams/', view=views.student_exams_tab, name='student-exams'),
    path('exams/take/<int:exam_id>/', view=views.student_take_exam, name='student-take-exam'),
    path('exams/results/<int:exam_id>/', view=views.student_view_results, name='student-view-results'),
    path('help/', view=views.student_help, name='student-help'),
]