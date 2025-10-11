# webhook_receiver/urls.py
from django.urls import path
from .views import (
    WebHookVerification, vectorize_database, chat_with_vectorized_db, health_check,
    CustomerListView, CustomerDashboardView, ContractListView, SupportTicketListView,
    dashboard_summary, customer_comprehensive_info
)

urlpatterns = [
    path('webhook/', WebHookVerification.as_view(), name='webhook'),
    path('health/', health_check, name='health_check'),
    # path('twilio-webhook/', views.twilio_webhook, name='twilio_webhook'),
    # path('send-message/', send_whatsapp_message, name='send_message'),  # Function not defined, commented out
    
    # New REST endpoints for database vectorization and chat
    path('vectorize-database/', vectorize_database, name='vectorize_database'),
    path('chat/', chat_with_vectorized_db, name='chat_with_db'),
    
    # Dashboard API endpoints
    path('dashboard/summary/', dashboard_summary, name='dashboard_summary'),
    path('dashboard/customers/', CustomerListView.as_view(), name='customer_list'),
    path('dashboard/customers/<int:id>/', CustomerDashboardView.as_view(), name='customer_dashboard'),
    path('dashboard/customers/<int:customer_id>/comprehensive/', customer_comprehensive_info, name='customer_comprehensive_info'),
    path('dashboard/contracts/', ContractListView.as_view(), name='contract_list'),
    path('dashboard/tickets/', SupportTicketListView.as_view(), name='ticket_list'),
]