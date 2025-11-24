from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect
from home.models import Profile, StudentProfile


class SignUpForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        label="Email"
    )
    birthday = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        label="Birthday"
    )
    LRN_number = forms.CharField(
        min_length=12,
        max_length=12,
        required=True,
        label="LRN Number"
    )
    first_name = forms.CharField(
        max_length=150,
        required=True,
        label="First Name"
    )
    last_name = forms.CharField(
        max_length=150,
        required=True,
        label="Last Name"
    )
    username = forms.CharField(
        max_length=70,
        required=True,
        label="Username"
    )
    role = forms.ChoiceField(
        choices=[('Teacher', 'Teacher'), ('Student', 'Student')],
        label="Role"
    )
    class_designation = forms.CharField(
        max_length=4,
        min_length= 4,
        required=True,
        widget=forms.TextInput(attrs={'placeholder': 'Type N/A if not a student'}),
        label="Class Designation",
        help_text="Type N/A if not a student."
    )

    class Meta:
        model = User
        fields = ('username', 'email', 'birthday', 'LRN_number', 'first_name', 'last_name' , 'password1', 'password2', 'role', 'class_designation')
    
    def clean_class_designation(self):
        # Validate class designation format
        class_designation = self.cleaned_data.get('class_designation', '').strip().upper()
        role = self.data.get('role')  # Get role from form data
        
        # If not a student, must be "N/A"
        if role == 'Teacher':
            if class_designation != 'N/A':
                raise forms.ValidationError(
                    "Teachers must enter 'N/A' for class designation."
                )
            return class_designation
        
        # If student, validate format
        if role == 'Student':
            if class_designation == 'N/A':
                raise forms.ValidationError(
                    "Students must provide a valid class designation (e.g., 0701, 1002)."
                )
            
            # Check if it's exactly 4 digits
            if len(class_designation) != 4 or not class_designation.isdigit():
                raise forms.ValidationError(
                    "Class designation must be exactly 4 digits (e.g., 0701, 1002)."
                )
            
            # Optional: Check if first 2 digits are valid grade levels (07-12)
            grade_level = int(class_designation[:2])
            if grade_level < 7 or grade_level > 12:
                raise forms.ValidationError(
                    "Invalid grade level. Must be between 07 and 12 (e.g., 0701, 1002)."
                )
            
            # Optional: Check if last 2 digits are valid section numbers (01-99)
            section = int(class_designation[2:])
            if section < 1 or section > 99:
                raise forms.ValidationError(
                    "Invalid section number. Must be between 01 and 99 (e.g., 0701, 1002)."
                )
        
        return class_designation
    
    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']

        if commit:
            user.save()
            # Save profile information
            Profile.objects.create(
                user=user,
                username=self.cleaned_data['username'],
                email=self.cleaned_data['email'],
                birthday=self.cleaned_data['birthday'],
                LRN_number=self.cleaned_data['LRN_number'],
                first_name=self.cleaned_data['first_name'],
                last_name=self.cleaned_data['last_name'],
                role=self.cleaned_data['role']
            )

            if self.cleaned_data['role'] == 'Student':
                StudentProfile.objects.create(
                    profile=user.profile,
                    class_designation=self.cleaned_data['class_designation']
                )
            elif self.cleaned_data['role'] == 'Teacher':
                pass  # Implement TeacherProfile creation if needed
                

        return user
    


class RoleBasedLoginView(LoginView):
    def get_success_url(self):
        user = self.request.user
        profile = getattr(user, 'profile', None)
        if profile:
            if profile.role == 'Student':
                return '/student'
            elif profile.role == 'Teacher':
                return '/teacher'
            elif profile.role == 'Admin':
                return '/admin'
        return '/'  # Default redirect