
from django.contrib import admin
from django.urls import path, include
from django.http import HttpResponse, JsonResponse
import requests
import os
import logging

def health(request):
    """Health check endpoint for Railway deployment"""
    try:
        # Check if Ollama is running
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        ollama_url = f"http://{ollama_host}/api/tags"
        
        response = requests.get(ollama_url, timeout=5)
        response.raise_for_status()
        
        # Check if our model is available
        models = response.json().get('models', [])
        model_names = [model.get('name', '') for model in models]
        
        has_llama = any('llama3.2:1b' in name for name in model_names)
        
        return JsonResponse({
            'status': 'healthy',
            'ollama_running': True,
            'ollama_host': ollama_host,
            'models_available': len(models),
            'llama_3_2_1b_available': has_llama,
            'available_models': model_names[:5]  # Limit to first 5 models
        }, status=200)
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama health check failed: {e}")
        return JsonResponse({
            'status': 'unhealthy',
            'ollama_running': False,
            'error': str(e)
        }, status=503)
    except Exception as e:
        logging.error(f"General health check error: {e}")
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/',include('webhook_receiver.urls')),
    path('health', health, name='health'),
]
