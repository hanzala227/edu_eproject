from django.urls import path
from analytics import views

urlpatterns = [
    path('dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('api/dashboard-data/', views.api_dashboard_data, name='api_dashboard_data'),
    path('api/sync-models/', views.api_sync_models, name='api_sync_models'),
]
