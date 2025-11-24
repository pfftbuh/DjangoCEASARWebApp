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
    