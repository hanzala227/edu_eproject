from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import check_password
from edupredict.db import get_db, COLL_USERS

def login_view(request):
    """
    Renders the modern login screen and authenticates users
    directly against the MongoDB Atlas `users` collection.
    """
    if request.session.get('user'):
        return redirect('admin_dashboard')

    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        if not username or not password:
            messages.error(request, "Please enter both username and password.")
            return render(request, 'accounts/login.html')

        try:
            db = get_db()
            user_doc = db['users'].find_one({'username': username})

            if user_doc and check_password(password, user_doc.get('password', '')):
                # Store user profile in Django session
                request.session['user'] = {
                    'username': user_doc['username'],
                    'full_name': user_doc.get('full_name', username),
                    'email': user_doc.get('email', ''),
                    'role': user_doc.get('role', 'student'),
                    'department': user_doc.get('department', 'General'),
                }
                messages.success(request, f"Welcome back, {user_doc.get('full_name', username)}!")
                return redirect('admin_dashboard')
            else:
                messages.error(request, "Invalid username or password. Please try again.")
        except Exception as e:
            messages.error(request, f"Database authentication error: {e}")

    return render(request, 'accounts/login.html')


def logout_view(request):
    """Logs the user out by clearing the session."""
    request.session.flush()
    messages.info(request, "You have been safely logged out.")
    return redirect('login')
