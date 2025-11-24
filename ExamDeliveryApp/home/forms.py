from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect
from home.models import Profile


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    birthday = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    LRN_number = forms.CharField(min_length=12, max_length=12, required=True)
    first_name = forms.CharField(max_length=150, required=True)
    last_name = forms.CharField(max_length=150, required=True)
    username = forms.CharField(max_length=70, required=True)
    role = forms.ChoiceField(
        choices=[('Teacher', 'Teacher'), ('Student', 'Student')])

    class Meta:
        model = User
        fields = ('username', 'email', 'birthday', 'LRN_number', 'first_name', 'last_name' , 'password1', 'password2', 'role')

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']

        if commit:
            user.save()
            # Save profile information
            from .models import Profile  # Import here to avoid circular import
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