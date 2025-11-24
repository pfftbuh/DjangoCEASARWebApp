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