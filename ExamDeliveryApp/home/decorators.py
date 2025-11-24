from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden
from home.models import Profile

def login_required_role(allowed_roles=[]):
    def decorator(view_func):
        @login_required
        def _wrapped_view(request, *args, **kwargs):
            try:
                profile = request.user.profile
                if profile.role in allowed_roles and profile.authorized:
                    return view_func(request, *args, **kwargs)
                else:
                    if not profile.authorized:
                        return HttpResponseForbidden("Your account is not authorized to access this page.")
                    elif profile.role not in allowed_roles:
                        return HttpResponseForbidden("You do not have permission to access this page.")
            except Profile.DoesNotExist:
                return HttpResponseForbidden("You do not have permission to access this page.")
        return _wrapped_view
    return decorator