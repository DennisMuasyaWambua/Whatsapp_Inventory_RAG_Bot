# webhook_receiver/urls.py
from django.urls import path
from .views import WebHookVerification

urlpatterns = [
    path('webhook/',WebHookVerification.as_view(), name='webhook'),
    # path('twilio-webhook/', views.twilio_webhook, name='twilio_webhook'),
    # path('send-message/', views.send_whatsapp_message, name='send_message'),
]