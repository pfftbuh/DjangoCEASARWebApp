import json
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from home.decorators import login_required_role
from home.models import Profile, StudentProfile, Exams, ExamSubmissions, ActiveExamSessions


@login_required_role(allowed_roles=['Student', 'Admin'])
def student_cam_calibration(request, exam_id):
    exam = Exams.objects.get(id=exam_id)
    student_profile = StudentProfile.objects.get(profile__user=request.user)

    # Check if student has remaining attempts
    case, _ = check_attempts(exam, student_profile)
    if not case:
        return HttpResponse("You have reached the maximum number of attempts for this exam.", status=400)
    
    # Reset calibration when entering calibration page
    from cameratrack.views import get_camera
    cam = get_camera()
    cam.reset_calibration()
    
    return render(request, 'studentside/cam_calibration.html', {'exam': exam})


def check_attempts(exam, student_profile):
    existing_submissions = ExamSubmissions.objects.filter(exam=exam.id, student=student_profile)
    attempt_number = existing_submissions.count()

    # Check if student has remaining attempts
    if exam.total_attempts > 0 and attempt_number >= exam.total_attempts:
        return False, -1
    else:
        return True, attempt_number + 1


def score_exam(student_answers, questions):
    
    score = 0.0
    
    # Parse student answers if it's a JSON string
    if isinstance(student_answers, str):
        try:
            student_answers = json.loads(student_answers)
        except:
            student_answers = {}
    
    # If student_answers is still empty or not a dict, return 0
    if not isinstance(student_answers, dict):
        return 0.0
    
    # Iterate through each question
    for question in questions:
        question_id = question.get('id')
        correct_answer = question.get('correct_answer')
        points = float(question.get('points', 1))  # Default to 1 point
        question_type = question.get('question_type')
        
        # Get student's answer for this question
        student_answer = student_answers.get(f'answer_{question_id}')
        
        if student_answer is None:
            continue  # No answer provided, skip
        
        # Compare based on question type
        if question_type == 'mcq_multi':
            # For multiple choice multiple answer, student_answer should be a list
            if isinstance(student_answer, list) and isinstance(correct_answer, list):
                # Check if sets match (order doesn't matter)
                if set(student_answer) == set(correct_answer):
                    score += points
            elif isinstance(student_answer, str) and isinstance(correct_answer, list):
                # If single answer given, convert to list
                if [student_answer] == correct_answer:
                    score += points
        
        elif question_type == 'numerical':
            # For numerical answers, allow small tolerance for floating point
            try:
                student_num = float(student_answer)
                correct_num = float(correct_answer)
                if abs(student_num - correct_num) < 0.001:  # Small tolerance
                    score += points
            except (ValueError, TypeError):
                pass  # Invalid number format
        
        elif question_type in ['true_false', 'mcq_one']:
            # For single answer questions, do exact string comparison
            if str(student_answer).strip() == str(correct_answer).strip():
                score += points
        
        else:
            # Generic comparison for other types
            if student_answer == correct_answer:
                score += points
    
    return round(score, 2)

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_dashboard(request):
    student_name = request.user.first_name
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    exams = Exams.objects.all()

    student_exams = [exam for exam in exams if student_profile.class_designation in exam.class_designation]

    # Calculate remaining attempts for each exam and attach to exam object
    for exam in student_exams:
        existing_submissions = ExamSubmissions.objects.filter(exam=exam.id, student=student_profile)
        attempt_number = existing_submissions.count()
        exam.attempts_left = exam.total_attempts - attempt_number if exam.total_attempts > 0 else 'Unlimited'

    return render(request, 'studentside/dashboard.html', {'student_name': student_name, 'exams': student_exams})

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_profile(request):
    return HttpResponse("This is the Student Profile Page")

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_exams_tab(request):
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    exams = Exams.objects.all()
    student_exams = [exam for exam in exams if student_profile.class_designation in exam.class_designation]
    return render(request, 'studentside/exams_tab.html', {'exams': student_exams})


def student_exam_details(request, exam_id):
    exam = Exams.objects.get(id=exam_id)
    case, attempt_number = check_attempts(exam, StudentProfile.objects.get(profile__user=request.user))
    isLocked = False
    
    if not case:
        isLocked = True
        attempts_left = 0
    else: 
        isLocked = False
        attempts_left = attempt_number - exam.total_attempts + 1
    
    return render(request, 'studentside/exam_details_page.html', {'exam': exam, 'isLocked': isLocked, 'attempts_left': attempts_left})

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_take_exam(request, exam_id):
    exam = Exams.objects.get(id=exam_id)
    student_profile = StudentProfile.objects.get(profile__user=request.user)

    # Check if student has remaining attempts
    case, _ = check_attempts(exam, student_profile)
    if not case:
        return HttpResponse("You have reached the maximum number of attempts for this exam.", status=400)
    
    else:

        # Get exam date and time limit
        exam_date = exam.exam_date
        time_limit = exam.time_limit

        # Check if date and time now is before or after exam date
        # from django.utils import timezone
        # now = timezone.now()
        # if exam_date and now < exam_date:
        #     return HttpResponse("The exam is not yet available. Please check the exam date and time.", status=403)
        # elif exam_date and now > exam_date + timezone.timedelta(minutes=time_limit):
        #     return HttpResponse("The exam time has passed. You can no longer take this exam.", status=403)

        # Set end time based on time now plus time limit
        from django.utils import timezone
        start_time = timezone.now()
        end_time = start_time + timezone.timedelta(minutes=time_limit)

        # Get the remaining time in seconds
        remaining_time = (end_time - start_time).total_seconds() // 60  # in minutes

        # Try to create an active exam session
        try:
            # If an active session already exists, do not create a new one
            active_session = ActiveExamSessions.objects.get(exam=exam, student=student_profile, is_active=True)
            start_time = active_session.start_time
            end_time = active_session.end_time  # Use existing end time
        except ActiveExamSessions.DoesNotExist:
            active_session = ActiveExamSessions.objects.create(
                exam=exam,
                student=student_profile,
                is_active=True,
                end_time=end_time  # Use newly calculated end_time
            )
        except Exception as e:
            return HttpResponse(f"Error creating active exam session: {str(e)}", status=500)

        # Get exam questions in a list
        questions = exam.questions
        print(f"DEBUG: Number of questions: {len(questions)}")
        print(f"DEBUG: Questions: {questions}")

        # Pass both time_limit and end_time to template
        return render(request, 'studentside/(debug)test_exam.html', {
            'exam': exam, 
            'questions': questions, 
            'exam_duration': time_limit,  # Duration in minutes
            'end_time': end_time.isoformat(),  # ISO format string for JavaScript
            'active_session': active_session
        })
    

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_submit_exam(request):
    if request.method == 'POST':
        exam_id = request.POST.get('exam_id')
        event_log = request.POST.get('event_log', '[]')
        
        if not exam_id:
            return HttpResponse("Error: Exam ID missing.", status=400)
        
        try:
            exam = Exams.objects.get(id=exam_id)
        except Exams.DoesNotExist:
            return HttpResponse("Error: Exam not found.", status=404)
        
        student_profile = StudentProfile.objects.get(profile__user=request.user)
        
        # Collect all answers from the form
        answers = {}
        for key, value in request.POST.items():
            if key.startswith('answer_'):
                # Extract question ID from key (e.g., 'answer_123' -> '123')
                question_id = key.replace('answer_', '')
                
                # Handle multiple checkbox values (for mcq_multi)
                if key in request.POST.getlist(key):
                    values = request.POST.getlist(key)
                    if len(values) > 1:
                        answers[key] = values  # Multiple values as list
                    else:
                        answers[key] = value  # Single value
                else:
                    answers[key] = value
        
        # Check if student has already submitted this exam and if they have attempts left
        case, attempt_number = check_attempts(exam, student_profile)
        if not case:
            return HttpResponse("You have reached the maximum number of attempts for this exam.", status=400)
        else:
            attempt_number = attempt_number
        
        # Calculate score by comparing answers
        score = score_exam(answers, exam.questions)
        
        # Create submission
        submission = ExamSubmissions.objects.create(
            exam=exam_id,
            student=student_profile,
            answers=json.dumps(answers),  # Store as JSON string
            event_log=event_log,
            attempt_number=attempt_number,
            score=score,
            behave_score=0,
        )

        submission.save()

        # Update active exam session to inactive
        try:
            active_session = ActiveExamSessions.objects.get(exam__id=exam_id, student=student_profile, is_active=True)
            active_session.is_active = False
            active_session.end_time = submission.submission_date
            active_session.save()
        except ActiveExamSessions.DoesNotExist:
            return HttpResponse("Active exam session not found.", status=404)
        
        # Place event log processing here if needed
        try:
            events = json.loads(event_log)
            key_count = 0
            altout_count = 0
            altin_count = 0
            copy_count = 0
            paste_count = 0

            for event in events: 
                ts = event.get('ts')
                if event.get('type') == 'keydown':
                    key_count += 1
                elif event.get('type') == 'focus':
                    altin_count += 1
                elif event.get('type') == 'blur':
                    altout_count += 1
                elif event.get('type') == 'copy':
                    copy_count += 1
                elif event.get('type') == 'paste':
                    paste_count += 1
            
            #List of count per event type
            events_list = {
                'Keypresses': key_count,
                'Alt Tabbed In': altin_count,
                'Alt Tabbed Out': altout_count,
                'Copied': copy_count,
                'Pasted': paste_count
            }

        except Exception as e:
            events = f"Error parsing event log: {str(e)}"
                
                
        return render(request, 'studentside/(debug)test_submit_exam.html', {'event_log': events_list, 'score': score})
        
    else:
        return HttpResponse("Invalid request method.", status=405)

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_grades_view(request):
    return render(request, 'studentside/grades_tab.html', {'score': 60})

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_view_results(request, exam_id):
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    submissions = ExamSubmissions.objects.filter(exam=exam_id, student=student_profile).order_by('-submission_date')
    
    if not submissions.exists():
        return HttpResponse("No submissions found for this exam.", status=404)
    
    latest_submission = submissions.first()
    exam = Exams.objects.get(id=exam_id)
    
    return render(request, 'studentside/view_results.html', {
        'submission': latest_submission,
        'exam': exam,
    })

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_help(request):
    return render(request, 'studentside/help.html')

## FOR TESTING PURPOSES ONLY
def violation_check(request, exam_id):
    user_id = request.user.id

    if request.method == 'POST':
        violation_detected = request.POST.get('violation') == 'true'
        if violation_detected:
            return HttpResponse(f"Violation logged by user {user_id} for exam {exam_id}", status=200)
        else:
            return HttpResponse("No violation", status=200)
    return HttpResponse("Invalid request", status=400)  


@login_required_role(allowed_roles=['Student', 'Admin'])
def teacher_start_exam(request, exam_id):

    user_id = request.user.id
    exam_id = request.GET.get('exam_id')  # Replace with actual exam_id from URL parameter
    
    # Logic to start the exam would go here
    if request.method == 'POST':
        # Process submitted event log
        event_log = request.POST.get('event_log', '[]')
        
    else:
        exam = Exams.objects.get(id=exam_id)
        # Get exam questions in a list
        questions = exam.questions
        return render(request, 'studentside/(debug)test_exam.html', {'exam': exam, 'questions': questions})
    
    # For debugging purposes
    if request.method == "GET":
        # Check if user has existing violations for this exam
        if user_id in exam_violations and exam_id in exam_violations.get(user_id, {}):
        # User has violations, redirect to dashboard
            from django.shortcuts import redirect
            return redirect('teacher-dashboard')
        if request.user.id in exam_violations:
            return HttpResponse("Violation logged", status=200)
        else:
            return render(request, 'teacherside/(debug)test_exam.html')

@login_required_role(allowed_roles=['Student', 'Admin'])
def student_active_exam(request, exam_id):
    # to update active exam session status
    student_profile = StudentProfile.objects.get(profile__user=request.user)
    try:
        active_session = ActiveExamSessions.objects.get(exam__id=exam_id, student=student_profile, is_active=True)
        return JsonResponse({'is_active': True, 'start_time': active_session.start_time})
    except ActiveExamSessions.DoesNotExist:
        return JsonResponse({'is_active': False})
