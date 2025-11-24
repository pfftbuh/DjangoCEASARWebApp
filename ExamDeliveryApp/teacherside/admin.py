from django.contrib import admin
from home.models import *

# Register your models here.
admin.site.register(Profile)
admin.site.register(StudentProfile)
admin.site.register(TeacherProfile)
admin.site.register(AdminProfile)
admin.site.register(ExamSubmissions)
admin.site.register(QuestionBanks)
admin.site.register(Exams)