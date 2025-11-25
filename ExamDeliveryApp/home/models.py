from django.db import models

# Create your models here.
class Profile(models.Model):
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE)
    username = models.CharField(max_length=150)
    email = models.EmailField()
    birthday = models.DateField()
    LRN_number = models.CharField(max_length=12, unique=True, default="000000000000")
    first_name = models.CharField(max_length=150, default="john")
    last_name = models.CharField(max_length=150, default="doe")
    ROLE_CHOICES = [
        ('Teacher', 'Teacher'),
        ('Student', 'Student'),
        ('Admin', 'Admin'),
    ]
    role = models.CharField(max_length=7, choices=ROLE_CHOICES)
    authorized = models.BooleanField(default=False)


    def __str__(self):
        return self.username
    
    class Meta:
        db_table = "Profiles"
        verbose_name = "Profile"
        verbose_name_plural = "Profiles"
    

class StudentProfile(models.Model):
    profile = models.OneToOneField(Profile, on_delete=models.CASCADE)
    class_designation = models.CharField(max_length=10, default="0701")

    def __str__(self):
        return f"{self.profile.username} - {self.class_designation}"
    
    class Meta:
        db_table = "Student Profiles"
        verbose_name = "Student Profile"
        verbose_name_plural = "Student Profiles"

## TO BE ALTERED TO PROPER FIELDS AND REQUIREMENTS ##
class TeacherProfile(models.Model):
    profile = models.OneToOneField(Profile, on_delete=models.CASCADE)
    subject_specialization = models.CharField(max_length=100, default="General")

    def __str__(self):
        return f"{self.profile.username} - {self.subject_specialization}"
    
    class Meta:
        db_table = "Teacher Profiles"
        verbose_name = "Teacher Profile"
        verbose_name_plural = "Teacher Profiles"

class AdminProfile(models.Model):
    profile = models.OneToOneField(Profile, on_delete=models.CASCADE)
    admin_level = models.CharField(max_length=50, default="Super Admin")

    def __str__(self):
        return f"{self.profile.username} - {self.admin_level}"
    
    class Meta:
        db_table = "Admin Profiles"
        verbose_name = "Admin Profile"
        verbose_name_plural = "Admin Profiles"

class Exams(models.Model):
    title = models.CharField(max_length=200)
    instructions = models.TextField()
    total_points = models.FloatField()
    total_attempts = models.IntegerField(default=1)
    questions = models.JSONField(default=list)
    exam_date = models.DateTimeField(null=True, blank=True)
    time_limit = models.IntegerField(help_text="Time limit in minutes", default=60)
    class_designation = models.JSONField(default=list, blank=True, help_text="List of class numbers")
    access_code = models.CharField(max_length=50, blank=True)
    access_code_required = models.BooleanField(default=False)
    question_count = models.IntegerField(default=0)
    date_created = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(Profile, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    
    class Meta:
        db_table = "Exams"
        verbose_name = "Exam"
        verbose_name_plural = "Exams"

class ExamSubmissions(models.Model):
    exam = models.IntegerField(default=1)
    attempt_number = models.IntegerField(default=1)
    answers = models.JSONField(default=list)
    event_log = models.JSONField(default=list, blank=True)
    behave_score = models.FloatField(default=0)
    student = models.ForeignKey(StudentProfile, on_delete=models.CASCADE, null=True, blank=True)
    score = models.FloatField()
    submission_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.profile.username} - {self.exam.title} - {self.score}"
    
    class Meta:
        db_table = "Exam Submissions"
        verbose_name = "Exam Submission"
        verbose_name_plural = "Exam Submissions"

class QuestionBanks(models.Model):
    questionbank_name = models.CharField(max_length=200, default="General Question Bank")
    question_and_answer = models.JSONField(default=list)
    created_by = models.ForeignKey(Profile, on_delete=models.CASCADE)

    def __str__(self):
        return f"Question Bank: {self.questionbank_name}"
    
    class Meta:
        db_table = "Question Banks"
        verbose_name = "Question Bank"
        verbose_name_plural = "Question Banks"
    