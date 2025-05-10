# webhook_receiver/serializers.py
from rest_framework import serializers
from .models import WhatsAppMessage

class WhatsAppMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = WhatsAppMessage
        fields = ['id', 'sender', 'recipient', 'message_type', 'content', 'media_id', 'timestamp', 'created_at']
        read_only_fields = ['created_at']

class SendMessageSerializer(serializers.Serializer):
    recipient = serializers.CharField(max_length=50)
    message = serializers.CharField()