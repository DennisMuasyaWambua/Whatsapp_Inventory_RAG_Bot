#!/usr/bin/env python3
"""
Health check endpoint for Railway deployment
This ensures the service is running and Ollama is accessible
"""

from flask import Flask, jsonify
import requests
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway"""
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
        
        return jsonify({
            'status': 'healthy',
            'ollama_running': True,
            'ollama_host': ollama_host,
            'models_available': len(models),
            'llama_3_2_1b_available': has_llama,
            'available_models': model_names[:5]  # Limit to first 5 models
        }), 200
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Ollama health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'ollama_running': False,
            'error': str(e)
        }), 503
    except Exception as e:
        logging.error(f"General health check error: {e}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available Ollama models"""
    try:
        ollama_host = os.getenv('OLLAMA_HOST', 'localhost:11434')
        ollama_url = f"http://{ollama_host}/api/tags"
        
        response = requests.get(ollama_url, timeout=5)
        response.raise_for_status()
        
        return response.json(), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)