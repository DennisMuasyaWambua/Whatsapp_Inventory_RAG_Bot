# webhook_receiver/urls.py
from django.urls import path
from .views import WebHookVerification, vectorize_database, chat_with_vectorized_db, health_check

urlpatterns = [
    path('webhook/', WebHookVerification.as_view(), name='webhook'),
    path('health/', health_check, name='health_check'),
    # path('twilio-webhook/', views.twilio_webhook, name='twilio_webhook'),
    # path('send-message/', send_whatsapp_message, name='send_message'),  # Function not defined, commented out
    
    # New REST endpoints for database vectorization and chat
    path('vectorize-database/', vectorize_database, name='vectorize_database'),
    path('chat/', chat_with_vectorized_db, name='chat_with_db'),
]