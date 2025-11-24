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
    description = models.TextField()
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    
    class Meta:
        db_table = "Exams"
        verbose_name = "Exam"
        verbose_name_plural = "Exams"

class ExamSubmissions(models.Model):
    exam = models.ForeignKey(Exams, on_delete=models.CASCADE)
    student = models.ForeignKey(StudentProfile, on_delete=models.CASCADE)
    score = models.FloatField()
    submission_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.student.profile.username} - {self.exam.title} - {self.score}"
    
    class Meta:
        db_table = "Exam Submissions"
        verbose_name = "Exam Submission"
        verbose_name_plural = "Exam Submissions"

class QuestionBanks(models.Model):
    question_text = models.TextField()
    correct_answer = models.TextField()

    def __str__(self):
        return f"Question for {self.exam.title}"
    
    class Meta:
        db_table = "Question Banks"
        verbose_name = "Question Bank"
        verbose_name_plural = "Question Banks"
    