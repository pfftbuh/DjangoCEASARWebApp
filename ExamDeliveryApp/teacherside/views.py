import json
from datetime import datetime
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from home.decorators import login_required_role
from home.models import Profile

class_roster = {
    '0701': ['Alice', 'Bob', 'Charlie'],
    '1002': ['David', 'Eve', 'Frank'],
    '0703': ['Grace', 'Heidi', 'Ivan']
}

exams = [
    {
        'id': 1,
        'title': 'Math Exam',
        'date': '2023-10-01 09:00',
        'duration': 60,
        'access_code_required': True,
        'access_code': 'MATH123',
        'instructions': 'Answer all questions. Calculators allowed.',
        'question_count': 3,
        'created_by': 'teacher1',
        'attempts': 1
    },
    {
        'id': 2,
        'title': 'Science Exam',
        'date': '2023-10-15 13:30',
        'duration': 90,
        'access_code_required': False,
        'access_code': '',
        'instructions': 'No reference materials allowed.',
        'question_count': 2,
        'created_by': 'teacher2',
        'attempts': 1
    },
    {
        'id': 3,
        'title': 'History Exam',
        'date': '2023-11-01 15:00',
        'duration': 45,
        'access_code_required': False,
        'access_code': '',
        'instructions': 'Write short answers for each question.',
        'question_count': 1,
        'created_by': 'teacher3',
        'attempts': 1
    },
]

examsubmissions = [
    {
        'id': 1,
        'exam_id': 1,
        'student_name': 'Alice',
        'score': 2,
        'answers': {'Q1': '4', 'Q2': '4'},
        'event_log': [],
        'submission_date': '2023-10-01 10:05',
        'attempt_number': 1,
        'behave_score': 10
    },
    {
        'id': 2,
        'exam_id': 1,
        'student_name': 'Bob',
        'score': 1,
        'answers': {'Q1': '3', 'Q2': '4'},
        'event_log': [],
        'submission_date': '2023-10-01 10:10',
        'attempt_number': 1,
        'behave_score': 20
    },
    {
        'id': 3,
        'exam_id': 2,
        'student_name': 'Charlie',
        'score': 1,
        'answers': {'Q1': 'Water'},
        'event_log': [],
        'submission_date': '2023-10-15 14:20',
        'attempt_number': 1,
        'behave_score': 30
    },
    {
        'id': 4,
        'exam_id': 3,
        'student_name': 'David',
        'score': 0,
        'answers': {'Q1': 'Lincoln'},
        'event_log': [],
        'submission_date': '2023-11-01 16:00',
        'attempt_number': 1,
        'behave_score': 40
    },
]

examquestions = [
    {
        'id': 1,
        'exam_id': 1,
        'question_number': 'Q1',
        'question_text': 'What is 2 + 2?',
        'correct_answer': '4',
        'points': 1,
        'question_type': 'numerical'
    },
    {
        'id': 2,
        'exam_id': 1,
        'question_number': 'Q2',
        'question_text': 'What is the square root of 16?',
        'correct_answer': '4',
        'points': 1,
        'question_type': 'numerical'
    },
    {
        'id': 3,
        'exam_id': 2,
        'question_number': 'Q1',
        'question_text': 'What is H2O commonly known as?',
        'correct_answer': 'Water',
        'points': 1,
        'question_type': 'multiple_choice_single',
        'choices': ['Water', 'Oxygen', 'Hydrogen', 'Salt']
    },
    {
        'id': 4,
        'exam_id': 3,
        'question_number': 'Q1',
        'question_text': 'Who was the first president of the USA?',
        'correct_answer': 'George Washington',
        'points': 1,
        'question_type': 'multiple_choice_single',
        'choices': ['Abraham Lincoln', 'George Washington', 'Thomas Jefferson', 'John Adams']
    },
    {
        'id': 5,
        'exam_id': 1,
        'question_number': 'Q3',
        'question_text': 'Select all prime numbers.',
        'correct_answer': ['2', '3', '5'],
        'points': 2,
        'question_type': 'multiple_choice_multiple',
        'choices': ['2', '3', '4', '5', '6']
    },
    {
        'id': 6,
        'exam_id': 2,
        'question_number': 'Q2',
        'question_text': 'The Earth is flat.',
        'correct_answer': 'False',
        'points': 1,
        'question_type': 'true_false'
    }
]

questionbanks = [
    {
        'id': 1,
        'subject': 'Math',
        'questions': [
            {'question': 'What is 2 + 2?', 'answer': '4'},
            {'question': 'What is the square root of 16?', 'answer': '4'},
            {'question': 'What is 10 divided by 2?', 'answer': '5'},
            {'question': 'What is 7 multiplied by 6?', 'answer': '42'},
            {'question': 'What is 15 minus 4?', 'answer': '11'},
            {'question': 'What is the value of pi (approx)?', 'answer': '3.14'},
        ]
    },
    {
        'id': 2,
        'subject': 'Science',
        'questions': [
            {'question': 'What is H2O commonly known as?', 'answer': 'Water'},
            {'question': 'What planet is known as the Red Planet?', 'answer': 'Mars'}
        ]
    },
    {
        'id': 3,
        'subject': 'History',
        'questions': [
            {'question': 'Who was the first president of the USA?', 'answer': 'George Washington'},
            {'question': 'In which year did World War II end?', 'answer': '1945'}
        ]
    },
]

exam_violations = {}  # Structure: {user_id: {exam_id: violation_count}}

def update_exam_question_count(exam_id):
    count = len([q for q in examquestions if q['exam_id'] == exam_id])
    for exam in exams:
        if exam['id'] == exam_id:
            exam['question_count'] = count
            break

def update_exam_scores(exam_id):
    score = 0 

    for examsubmission in examsubmissions:
        if examsubmission['exam_id'] == exam_id:
            score = 0  # Reset score for each submission

            for question in examquestions:
                if question['exam_id'] == exam_id:
                    qnum = question['question_number']
                    if qnum in examsubmission.get('answers', {}):
                        student_answer = examsubmission['answers'][qnum]
                        correct_answer = question['correct_answer']
                        # Handle multiple correct answers for multiple_choice_multiple
                        if question['question_type'] == 'multiple_choice_multiple':
                            # Both should be lists of strings
                            if set(student_answer) == set(correct_answer):
                                score += question['points']
                        else:
                            if str(student_answer).strip() == str(correct_answer).strip():
                                score += question['points']
                    else:
                        # If the answer is missing, no points are added
                        pass

            examsubmission['score'] = score





def update_exam_details(exam_id, exam_duration, exam_instructions, access_code_required, access_code, attempts, exam_date):
    for exam in exams:
        if exam['id'] == exam_id:
            exam['duration'] = exam_duration
            exam['instructions'] = exam_instructions
            exam['access_code_required'] = access_code_required
            exam['access_code'] = access_code if access_code_required else ''
            exam['attempts'] = attempts
            exam_date = exam_date.strip().replace("T", " ")
            exam['date'] = exam_date
            break



@login_required_role(allowed_roles=['Teacher', 'Admin'])
def get_question_banks_api(request):
    """API endpoint to return question banks as JSON"""
    return JsonResponse({'question_banks': questionbanks}, safe=False)



# Create your views here.
@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_dashboard(request):
    # Get the user's profile to retrieve first and last name
    profile = Profile.objects.filter(user=request.user).first()
    teacher_name = f"{profile.first_name} {profile.last_name}" if profile else request.user.username
    return render(request, 'teacherside/dashboard.html', context={'exams': exams, 'teacher_name': teacher_name})





@login_required_role(allowed_roles=['Teacher','Admin'])
def teacher_delete_question(request, exam_id, question_id):
    if request.method == 'POST':
        # Find and remove the question
        global examquestions
        examquestions = [q for q in examquestions if not (q['id'] == question_id and q['exam_id'] == exam_id)]
        
        # Update question numbers for remaining questions in this exam
        remaining_questions = [q for q in examquestions if q['exam_id'] == exam_id]
        for index, question in enumerate(remaining_questions, start=1):
            question['question_number'] = f'Q{index}'
        
        # Update the exam question count and scores
        update_exam_question_count(exam_id)
        update_exam_scores(exam_id)
        
        # Redirect back to modify exam page
        from django.shortcuts import redirect
        return redirect('teacher-modify-exam', exam_id=exam_id)
    
    # If not POST, redirect to modify exam page
    from django.shortcuts import redirect

    return redirect('teacher-modify-exam', exam_id=exam_id)






@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_profile(request):
    return render(request, 'teacherside/profile.html')





@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_manage_exams(request):
    return render(request, 'teacherside/manage_exams.html', {'exams': exams})




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

        # Save the new question
        new_question = {
            'id': len(examquestions) + 1,
            'exam_id': exam_id,
            'question_number': f'Q{len([q for q in examquestions if q["exam_id"] == exam_id]) + 1}',
            'question_text': question_text,
            'correct_answer': correct_answer,  # Will be a list for mcq_multi, string for others
            'points': points,
            'choices': choices,
            'question_type': question_type
        }

        examquestions.append(new_question)
        update_exam_question_count(exam_id)
        update_exam_scores(exam_id)
        exam = next((exam for exam in exams if exam['id'] == exam_id), None)
        new_examquestions = [q for q in examquestions if q['exam_id'] == exam_id]
        return render(request, 'teacherside/modify_exam.html', {'exam': exam, 'existing_questions': new_examquestions, 'question_bank': questionbanks})
    else:
        exam = next((exam for exam in exams if exam['id'] == exam_id), None)
        existing_questions = [q for q in examquestions if q['exam_id'] == exam_id]
        if not exam:
            return HttpResponse("Exam not found", status=404)
        return render(request, 'teacherside/modify_exam.html', {'exam': exam, 'question_bank': questionbanks, 'existing_questions': existing_questions})






@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_create_exam(request):
    if request.method == 'POST':
        # Process form data here
        exam_title = request.POST.get('exam_name')
        exam_duration = request.POST.get('duration')
        exam_date = request.POST.get('scheduled_time')
        exam_instructions = request.POST.get('instructions')
        access_code_required = request.POST.get('access_code_required') == 'on'
        exam_attempts = int(request.POST.get('num_attempts', 1))
        selected_class_id = request.POST.get('class_list')
        # Save the new exam (this is just a placeholder, implement actual saving logic)
        new_exam = {
            'id': len(exams) + 1,
            'title': exam_title,
            'date': exam_date,
            'duration' : exam_duration,
            'instructions': exam_instructions,
            'question_count': 0,
            'access_code_required': access_code_required,
            'access_code': request.POST.get('access_code') if access_code_required else '',
            'attempts': exam_attempts,
            'created_by': request.user.username,
        }
        exams.append(new_exam)
        return render(request, 'teacherside/create_exam_success.html', {'exam': new_exam})
    
    return render(request, 'teacherside/create_exam.html', {'classes': class_roster})






@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_view_submissions(request, exam_id):
    exam = next((exam for exam in exams if exam['id'] == exam_id), None)
    submissions = [sub for sub in examsubmissions if sub['exam_id'] == exam_id]
    return render(request, 'teacherside/view_submissions.html', {'exam': exam, 'submissions': submissions})






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



## FOR TESTING PURPOSES ONLY
def violation_check(request):
    user_id = request.user.id
    exam_id = 1  # Replace with actual exam_id from URL parameter

    if request.method == 'POST':
        violation_detected = request.POST.get('violation') == 'true'
        if violation_detected:
            if user_id not in exam_violations:
                exam_violations[user_id] = {}
            if exam_id not in exam_violations[user_id]:
                exam_violations[user_id][exam_id] = 0
            exam_violations[user_id][exam_id] += 1
            return HttpResponse("Violation logged", status=200)
        else:
            return HttpResponse("No violation", status=200)
    return HttpResponse("Invalid request", status=400)  


@login_required_role(allowed_roles=['Teacher', 'Admin'])
def teacher_start_exam(request):

    user_id = request.user.id
    exam_id = 1  # Replace with actual exam_id from URL parameter
    
    

    # Logic to start the exam would go here
    if request.method == 'POST':
        # Process submitted event log
        event_log = request.POST.get('event_log', '[]')
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
                
                
        return render(request, 'teacherside/(debug)test_submit_exam.html', {'event_log': events_list})
    
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
    
    