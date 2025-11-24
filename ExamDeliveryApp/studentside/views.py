from django.shortcuts import render
from django.http import HttpResponse
from home.decorators import login_required_role


sections_with_students = {
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
        'attempts': 1,
        'class_assigned': '0701'  # Alice, Bob, Charlie
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
        'attempts': 1,
        'class_assigned': '0701'  # Charlie is in 0701, so align to 0701
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
        'attempts': 1,
        'class_assigned': '1002'  # David is in 1002
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
