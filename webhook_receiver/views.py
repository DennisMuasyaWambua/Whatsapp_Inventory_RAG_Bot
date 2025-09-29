# webhook_receiver/views.py
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import logging

from webhook_receiver.chat import chat_with_database, create_vector_store_from_db
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


@api_view(['POST'])
def vectorize_database(request):
    """
    REST endpoint to vectorize any database dynamically.
    
    Expected POST data:
    {
        "db_url": "postgresql://user:pass@localhost:5432/dbname",
        "embedding_model": "all-MiniLM-L6-v2" (optional)
    }
    """
    try:
        data = request.data
        db_url = data.get('db_url')
        
        if not db_url:
            return Response(
                {"error": "db_url is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        embedding_model = data.get('embedding_model', 'all-MiniLM-L6-v2')
        
        logging.info(f"Starting vectorization for database: {db_url}")
        
        # Create vector store from the provided database
        vector_store = create_vector_store_from_db(db_url, embedding_model)
        
        logging.info("Database vectorization completed successfully")
        
        return Response({
            "message": "Database vectorized successfully",
            "vector_store_path": "faiss_index_store",
            "embedding_model": embedding_model
        }, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        logging.error(f"Error vectorizing database: {str(e)}")
        return Response(
            {"error": f"Failed to vectorize database: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def chat_with_vectorized_db(request):
    """
    REST endpoint to chat with a vectorized database.
    
    Expected POST data:
    {
        "query": "What products do you have?",
        "db_url": "postgresql://user:pass@localhost:5432/dbname" (optional, uses settings if not provided)
    }
    """
    try:
        data = request.data
        query = data.get('query')
        
        if not query:
            return Response(
                {"error": "query is required"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Use provided db_url or fall back to settings
        from django.conf import settings
        db_url = data.get('db_url', getattr(settings, 'DB_URL', None))
        
        if not db_url:
            return Response(
                {"error": "db_url must be provided or configured in settings"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        logging.info(f"Processing chat query: {query}")
        
        # Get response from chat function
        response_text = chat_with_database(db_url, query)
        
        logging.info("Chat query processed successfully")
        
        return Response({
            "query": query,
            "response": response_text
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logging.error(f"Error processing chat query: {str(e)}")
        return Response(
            {"error": f"Failed to process query: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )