from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

def root_redirect(request):
    if request.session.get('user'):
        return redirect('admin_dashboard')
    return redirect('login')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', root_redirect, name='root'),
    path('auth/', include('accounts.urls')),
    path('', include('accounts.urls')),
    path('', include('analytics.urls')),
]
