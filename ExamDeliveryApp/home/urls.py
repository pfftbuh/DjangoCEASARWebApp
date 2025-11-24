from django.urls import path
from django.contrib.auth import views as auth_views
from home import views as home_views
from home.forms import RoleBasedLoginView

urlpatterns = [
    path('', view=home_views.home_page, name='home-page'),
    path('about/', view=home_views.about_page, name='about-page'),
    path('contact/', view=home_views.contact_page, name='contact-page'),
    path('site_login/', RoleBasedLoginView.as_view(template_name='home/site_login.html'), name='site-login'),
    path('site_signup/', view=home_views.site_signup, name='site-signup'),
    path('logout/', auth_views.LogoutView.as_view(template_name='home/site-logout.html'), name='site-logout'),
]