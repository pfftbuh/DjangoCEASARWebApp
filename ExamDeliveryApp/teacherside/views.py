import json
from datetime import datetime
from sqlite3 import IntegrityError
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from home.decorators import login_required_role
from home.models import Profile, TeacherProfile, Exams, StudentProfile, QuestionBanks, ExamSubmissions

# class_roster = {
#     '0701': ['Alice', 'Bob', 'Charlie'],
#     '1002': ['David', 'Eve', 'Frank'],
#     '0703': ['Grace', 'Heidi', 'Ivan']
# }

# exams = [
#     {
#         'id': 1,
#         'title': 'Math Exam',
#         'date': '2023-10-01 09:00',
#         'duration': 60,
#         'access_code_required': True,
#         'access_code': 'MATH123',
#         'instructions': 'Answer all questions. Calculators allowed.',
#         'question_count': 3,
#         'created_by': 'teacher1',
#         'attempts': 1
#     },
#     {
#         'id': 2,
#         'title': 'Science Exam',
#         'date': '2023-10-15 13:30',
#         'duration': 90,
#         'access_code_required': False,
#         'access_code': '',
#         'instructions': 'No reference materials allowed.',
#         'question_count': 2,
#         'created_by': 'teacher2',
#         'attempts': 1
#     },
#     {
#         'id': 3,
#         'title': 'History Exam',
#         'date': '2023-11-01 15:00',
#         'duration': 45,
#         'access_code_required': False,
#         'access_code': '',
#         'instructions': 'Write short answers for each question.',
#         'question_count': 1,
#         'created_by': 'teacher3',
#         'attempts': 1
#     },
# ]

# examsubmissions = [
#     {
#         'id': 1,
#         'exam_id': 1,
#         'student_name': 'Alice',
#         'score': 2,
#         'answers': {'Q1': '4', 'Q2': '4'},
#         'event_log': [],
#         'submission_date': '2023-10-01 10:05',
#         'attempt_number': 1,
#         'behave_score': 10
#     },
#     {
#         'id': 2,
#         'exam_id': 1,
#         'student_name': 'Bob',
#         'score': 1,
#         'answers': {'Q1': '3', 'Q2': '4'},
#         'event_log': [],
#         'submission_date': '2023-10-01 10:10',
#         'attempt_number': 1,
#         'behave_score': 20
#     },
#     {
#         'id': 3,
#         'exam_id': 2,
#         'student_name': 'Charlie',
#         'score': 1,
#         'answers': {'Q1': 'Water'},
#         'event_log': [],
#         'submission_date': '2023-10-15 14:20',
#         'attempt_number': 1,
#         'behave_score': 30
#     },
#     {
#         'id': 4,
#         'exam_id': 3,
#         'student_name': 'David',
#         'score': 0,
#         'answers': {'Q1': 'Lincoln'},
#         'event_log': [],
#         'submission_date': '2023-11-01 16:00',
#         'attempt_number': 1,
#         'behave_score': 40
#     },
# ]

# examquestions = [
#     {
#         'id': 1,
#         'exam_id': 1,
#         'question_number': 'Q1',
#         'question_text': 'What is 2 + 2?',
#         'correct_answer': '4',
#         'points': 1,
#         'question_type': 'numerical'
#     },
#     {
#         'id': 2,
#         'exam_id': 1,
#         'question_number': 'Q2',
#         'question_text': 'What is the square root of 16?',
#         'correct_answer': '4',
#         'points': 1,
#         'question_type': 'numerical'
#     },
#     {
#         'id': 3,
#         'exam_id': 2,
#         'question_number': 'Q1',
#         'question_text': 'What is H2O commonly known as?',
#         'correct_answer': 'Water',
#         'points': 1,
#         'question_type': 'multiple_choice_single',
#         'choices': ['Water', 'Oxygen', 'Hydrogen', 'Salt']
#     },
#     {
#         'id': 4,
#         'exam_id': 3,
#         'question_number': 'Q1',
#         'question_text': 'Who was the first president of the USA?',
#         'correct_answer': 'George Washington',
#         'points': 1,
#         'question_type': 'multiple_choice_single',
#         'choices': ['Abraham Lincoln', 'George Washington', 'Thomas Jefferson', 'John Adams']
#     },
#     {
#         'id': 5,
#         'exam_id': 1,
#         'question_number': 'Q3',
#         'question_text': 'Select all prime numbers.',
#         'correct_answer': ['2', '3', '5'],
#         'points': 2,
#         'question_type': 'multiple_choice_multiple',
#         'choices': ['2', '3', '4', '5', '6']
#     },
#     {
#         'id': 6,
#         'exam_id': 2,
#         'question_number': 'Q2',
#         'question_text': 'The Earth is flat.',
#         'correct_answer': 'False',
#         'points': 1,
#         'question_type': 'true_false'
#     }
# ]

exam_violations = {}  # Structure: {user_id: {exam_id: violation_count}}

def update_exam_question_count(exam_id):
    exam = Exams.objects.filter(id=exam_id).first()
    count = len(exam.questions) if exam and exam.questions else 0
    if exam:
        exam.question_count = count
        exam.save()

def update_exam_scores(exam_id):

    exam = Exams.objects.filter(id=exam_id).first()
    if not exam or not exam.questions:
        return

    questions = exam.questions

    # Build a dict for quick lookup: {question_number: question_dict}
    question_map = {q['question_number']: q for q in questions if 'question_number' in q}

    # Update all submissions for this exam
    submissions = ExamSubmissions.objects.filter(exam=exam_id)
    for submission in submissions:
        answers = submission.answers or {}
        score = 0

        for qnum, question in question_map.items():
            if qnum in answers:
                student_answer = answers[qnum]
                correct_answer = question.get('correct_answer')
                qtype = question.get('question_type')
                points = question.get('points', 0)

                if qtype == 'multiple_choice_multiple':
                    # Both should be lists
                    # Partial credit: count how many student answers are in correct_answer
                    if isinstance(student_answer, list) and isinstance(correct_answer, list):
                        match_count = sum(1 for ans in student_answer if ans in correct_answer)
                        if len(correct_answer) > 0:
                            score += (match_count / len(correct_answer)) * points
                else:
                    if str(student_answer).strip() == str(correct_answer).strip():
                        score += points

        submission.score = score
        submission.save()



def update_exam_details(exam_id, exam_duration, exam_instructions, access_code_required, access_code, attempts, exam_date):
    try:
        exam = Exams.objects.get(id=exam_id)
        exam.time_limit = int(exam_duration) if exam_duration else exam.time_limit
        exam.instructions = exam_instructions
        exam.access_code_required = access_code_required
        exam.access_code = access_code if access_code_required else ''
        exam.total_attempts = attempts
        if exam_date:
            # Handle both string and datetime input
            if isinstance(exam_date, str):
                exam.exam_date = datetime.strptime(exam_date.strip().replace("T", " "), "%Y-%m-%d %H:%M")
            else:
                exam.exam_date = exam_date
        exam.save()
    except Exams.DoesNotExist:
        pass





@login_required_role(allowed_roles=['Teacher', 'Admin'])
def get_question_banks_api(request):
    # API endpoint to return question banks as JSON
    questionbanks = QuestionBanks.objects.all()
    questionbanks_list = []

    # Debug: Print the actual structure
    

    for bank in questionbanks:
        qa_pairs = bank.question_and_answer if bank.question_and_answer else []
        
        print(f"Bank Name: {bank.questionbank_name}")
        print(f"QA Pairs Type: {type(bank.question_and_answer)}")
        print(f"QA Pairs Content: {bank.question_and_answer}")
        # Create structured questions list with both question and answer
        questions = []
        for qa in qa_pairs:
            questions.append({
                'question': qa.get('question', ''),
                'answer': qa.get('answer', '')
            })

        questionbanks_list.append({
            'id': bank.id,
            'questionbank_name': bank.questionbank_name,
            'questions': questions  # Array of {question, answer} objects
        })

        print(f"Processed Questions: {questions}")

    return JsonResponse({'question_banks': questionbanks_list}, safe=False)



@login_required_role(allowed_roles=['Teacher','Admin'])
def update_specialization(request):
    if request.method == 'POST':
        specialization = request.POST.get('specialization')
        profile = Profile.objects.filter(user=request.user).first()
        if not profile:
            return HttpResponse("Profile not found", status=404)
        teacher_profile = TeacherProfile.objects.filter(profile=profile).first()
        if not teacher_profile:
            return HttpResponse("Teacher profile not found", status=404)
        teacher_profile.subject_specialization = specialization
        teacher_profile.save()
        return redirect('teacher-profile')
    return HttpResponse("Error in handling data. Sorry! Please try again!", status=404)




# Create your views here.
@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_dashboard(request):
    # Get the user's profile to retrieve first and last name
    profile = Profile.objects.filter(user=request.user).first()
    teacher_name = f"{profile.first_name} {profile.last_name}" if profile else ""
    teacher_exams = Exams.objects.all()
    return render(request, 'teacherside/dashboard.html', context={'exams': teacher_exams, 'teacher_name': teacher_name})





@login_required_role(allowed_roles=['Teacher','Admin'])
def teacher_delete_question(request, exam_id, question_id):
    if request.method == 'POST':
        # Find and remove the question
        exam = Exams.objects.filter(id=exam_id).first()
        if not exam:
            return HttpResponse("Exam not found", status=404)
        questions = exam.questions if exam.questions else []
        questions = [q for q in questions if q['id'] != question_id]
        exam.questions = questions
        exam.save()
        
        # Update the exam question count and scores
        update_exam_question_count(exam_id)
        update_exam_scores(exam_id)
        
        # Redirect back to modify exam page
        return redirect('teacher-modify-exam', exam_id=exam_id)
    
    # If not POST, redirect to modify exam page

    return redirect('teacher-modify-exam', exam_id=exam_id)


@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_delete_exam(request, exam_id):
    if request.method == 'POST':
        exam = Exams.objects.filter(id=exam_id).first()
        if exam:
            exam.delete()
            return redirect('teacher-manage-exams')
        else:
            return HttpResponse("Exam not found", status=404)
    return HttpResponse("Invalid request method", status=400)


@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_profile(request):
    profile = Profile.objects.filter(user=request.user).first()
    teacher_profile = TeacherProfile.objects.filter(profile=profile).first()
    specialization = teacher_profile.subject_specialization if teacher_profile else ''
    return render(request, 'teacherside/teacher_profile.html', {'specialization': specialization})




@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_manage_exams(request):
    teacher_exams = Exams.objects.all()
    return render(request, 'teacherside/manage_exams.html', {'exams': teacher_exams})




@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_modify_exam_details(request, exam_id):
    if request.method == 'POST':
        # Process form data here
        exam_duration = request.POST.get('duration')
        exam_instructions = request.POST.get('instructions')
        access_code_required = request.POST.get('access_code_required') == 'on'
        access_code = request.POST.get('access_code') if access_code_required else ''
        attempts = int(request.POST.get('num_attempts', 1))
        exam_date = request.POST.get('scheduled_time')
        
        # Update exam details
        update_exam_details(exam_id, exam_duration, exam_instructions, access_code_required, access_code, attempts, exam_date)
        
        exam = next((exam for exam in exams if exam['id'] == exam_id), None)
        return render(request, 'teacherside/modify_exam.html', {'exam': exam, 'question_bank': questionbanks})
    else:
        exam = next((exam for exam in exams if exam['id'] == exam_id), None)
        if not exam:
            return HttpResponse("Exam not found", status=404)
        return render(request, 'teacherside/modify_exam.html', {'exam': exam, 'question_bank': questionbanks})



@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_modify_exam(request, exam_id):
    if request.method == 'POST':
        # Process form data here
        question_text = request.POST.get('questions')
        points = int(request.POST.get('points', 1))
        question_type = request.POST.get('question_type')
        
        # Collect all choices for multiple choice questions
        choices = []
        if question_type in ['mcq_one', 'mcq_multi']:
            choices = request.POST.getlist('choices')
        elif question_type == 'true_false':
            choices = ['True', 'False']
        
        # Handle correct answers based on question type
        if question_type == 'mcq_multi':
            # For multiple choice with multiple answers, get all checked checkboxes
            correct_answer = request.POST.getlist('correct_answer')
        else:
            # For single answer types (mcq_one, true_false, numerical)
            correct_answer = request.POST.get('correct_answer')

        exam = Exams.objects.filter(id=exam_id).first()

        if not exam:
            return HttpResponse("Exam not found", status=404)
        elif exam:
            # Prepare the new question dict to be appended to the Exam.questions JSONField
            new_question = {
                'id': len(exam.questions) + 1,
                'exam_id': exam_id,
                'question_number': f'Q{len(exam.questions) + 1}',
                'question_text': question_text,
                'correct_answer': correct_answer,  # Will be a list for mcq_multi, string for others
                'points': points,
                'choices': choices,
                'question_type': question_type
            }

            questions = exam.questions if exam.questions else []
            questions.append(new_question)
            exam.questions = questions
            exam.question_count = len(questions)
            exam.save()
        else:
            return HttpResponse("Exam not found", status=404)
        
        # Fetch the updated exam from the database
        exam = Exams.objects.filter(id=exam_id).first()
        existing_questions = exam.questions if exam and exam.questions else []
        question_bank = list(QuestionBanks.objects.all())
        return render(request, 'teacherside/modify_exam.html', {
            'exam': exam,
            'existing_questions': existing_questions,
            'question_bank': question_bank
        })
    
    exam = Exams.objects.filter(id=exam_id).first()
    existing_questions = exam.questions if exam and exam.questions else []
    question_bank = list(QuestionBanks.objects.all())

    # Get all unique classes from student profiles
    student_classes = list(
        StudentProfile.objects.values_list('class_designation', flat=True).distinct()
    )

    return render(
        request,
        'teacherside/modify_exam.html',
        {
            'exam': exam,
            'question_bank': question_bank,
            'existing_questions': existing_questions,
            'classes': student_classes
        }
    )


@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_create_exam(request):
    if request.method == 'POST':
        try:
            profile = Profile.objects.filter(user=request.user).first()
            exam_date = request.POST.get('scheduled_time')
            exam_title = request.POST.get('exam_name')
            exam_duration = int(request.POST.get('duration', 60))
            exam_instructions = request.POST.get('instructions')
            access_code_required = request.POST.get('access_code_required') == 'on'
            exam_attempts = int(request.POST.get('num_attempts', 1))
            selected_class_id = request.POST.get('class_list')
            
            # Create new exam in database
            new_exam = Exams.objects.create(
                title=exam_title,
                exam_date=exam_date,
                instructions=exam_instructions,
                total_points=0,  # Will be calculated as questions are added
                total_attempts=exam_attempts,
                questions=[],  # Empty list initially
                time_limit=exam_duration,
                class_designation=[selected_class_id] if selected_class_id else [],
                access_code=request.POST.get('access_code') if access_code_required else '',
                access_code_required=access_code_required,
                question_count=0,
                created_by=profile
            )
        except IntegrityError as e:
            return HttpResponse (f"You not have your teacher profile setup yet. {e}")
        except Exception as e:
            return HttpResponse(f"An error occurred: {e}", status=500)
        
        return render(request, 'teacherside/create_exam_success.html', {'exam': new_exam})
    
    # Get all unique classes from student profiles
    student_classes = list(
        StudentProfile.objects.values_list('class_designation', flat=True).distinct()
    )
    return render(request, 'teacherside/create_exam.html', {'classes': student_classes})






@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_view_submissions(request, exam_id):
    exam_submissions = ExamSubmissions.objects.filter(exam=exam_id)
    exam = Exams.objects.filter(id=exam_id).first()
    return render(request, 'teacherside/view_submissions.html', {'exam': exam, 'submissions': exam_submissions})






@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_grade_submission(request, submission_id):
    submission = next((sub for sub in examsubmissions if sub['id'] == submission_id), None)
    if submission:
        exam = next((exam for exam in exams if exam['id'] == submission['exam_id']), None)
        questions = [q for q in examquestions if q['exam_id'] == submission['exam_id']]
        
        # Prepare detailed answer info
        answer_details = []
        for q in questions:
            student_answer = submission['answers'].get(q['question_number'], '')
            correct_answer = q['correct_answer']
            question_text = q['question_text']
            points = q['points']
            answer_details.append({
                'question': question_text,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'points': points,
            })
        
        
        return render(request, 'teacherside/view_submission_detail.html', {
            'submission': submission, 
            'exam': exam, 
            'questions': questions,
            'answer_details': answer_details
        })
    return HttpResponse("Submission not found", status=404)

@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_generate_reports(request):
    return render(request, 'teacherside/generate_reports.html')

@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_help(request):
    return render(request, 'teacherside/help.html')


@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_violation_logs(request):
    return render(request, 'teacherside/violation_logs.html')




    