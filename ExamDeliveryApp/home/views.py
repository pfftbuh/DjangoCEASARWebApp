from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm
from .forms import SignUpForm
from django.contrib import messages
from home.models import Profile
from django.contrib.auth import logout

# Create your views here.
def home_page(request):
    return render(request, 'home/home.html')

def about_page(request):
    return render(request, 'home/about.html')

def contact_page(request):
    return render(request, 'home/contact.html')

def site_signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Account created successfully!')
            return redirect('home-page')
    else:
        form = SignUpForm()
    return render(request, 'home/site_signup.html', {'form': form})







