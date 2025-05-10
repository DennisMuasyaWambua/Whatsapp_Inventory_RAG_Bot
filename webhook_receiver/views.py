# webhook_receiver/views.py
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from webhook_receiver.chat import chat_with_database
from webhook_receiver.utils import verify, handle_message

class WebHookVerification(APIView):
     def get(self, request):
          # Add more logging for debugging
          import logging
          logging.info(f"GET request to webhook with query params: {dict(request.GET.items())}")
          
          result, status_code = verify(request)
          
          # Log the result
          logging.info(f"Webhook verification result: {result}, status: {status_code}")
          
          if isinstance(result, str):
               # If it's a challenge string, return it directly
               return HttpResponse(result, status=status_code)
          # Otherwise it's a dict that needs to be wrapped in Response
          return Response(result, status=status_code)
          
     def post(self, request):
          # Add more logging for debugging
          import logging
          logging.info(f"POST request to webhook")
          
          message, status_code = handle_message(request)

          print(request.body)
          
          # Log the result
          logging.info(f"Message handling result: {message}, status: {status_code}")
          
          return Response(message, status=status_code)