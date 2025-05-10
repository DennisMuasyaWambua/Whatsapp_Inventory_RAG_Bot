# webhook_receiver/models.py
from django.db import models

class WhatsAppMessage(models.Model):
    sender = models.CharField(max_length=50)
    recipient = models.CharField(max_length=50)
    message_type = models.CharField(max_length=20)
    content = models.TextField(blank=True)
    media_id = models.CharField(max_length=255, blank=True)
    timestamp = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Message from {self.sender} to {self.recipient}"    