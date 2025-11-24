from django.shortcuts import render
from django.http import HttpResponse
from home.decorators import login_required_role


@login_required_role(allowed_roles=['Student', 'Admin'])
def student_dashboard(request):
    return render(request, 'studentside/dashboard.html')

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_profile(request):
    return HttpResponse("This is the Student Profile Page")

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_exams_tab(request):
    return HttpResponse("Here are your courses")

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_take_exam(request, exam_id):
    return HttpResponse(f"Taking exam with ID: {exam_id}")

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_view_results(request, exam_id):
    return HttpResponse(f"Viewing results for exam with ID: {exam_id}")

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_help(request):
    return render(request, 'studentside/help.html')
