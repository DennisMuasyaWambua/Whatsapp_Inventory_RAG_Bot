# webhook_receiver/views.py
from django.http import HttpResponse, JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
import logging
import requests
import os

from webhook_receiver.chat import chat_with_database, create_vector_store_from_db
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_limiter import check_system_resources
from webhook_receiver.utils import verify, handle_message
from webhook_receiver.memory_monitor import monitor_memory, MemoryLimitedProcessor
from .models import Customer, Contract, SupportTicket, Product
from .seriailzers import CustomerDashboardSerializer, CustomerSerializer, ContractSerializer, SupportTicketSerializer
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.pagination import PageNumberPagination
from django.db import models

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
@monitor_memory
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
        
        embedding_model = data.get('embedding_model', 'paraphrase-MiniLM-L3-v2')
        
        logging.info(f"Starting vectorization for database: {db_url}")
        
        # Use memory-limited processor for vectorization
        with MemoryLimitedProcessor(memory_limit_mb=1024, cleanup_threshold_mb=768):
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
        
        # Check system resources before processing
        can_continue, resource_message = check_system_resources()
        if not can_continue:
            logging.warning(f"System resources insufficient: {resource_message}")
            return Response(
                {"error": f"System resources insufficient: {resource_message}. Please close other applications and try again."}, 
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        # Add memory monitoring before processing
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        logging.info(f"Memory before processing: {initial_memory:.2f} MB")
        
        # Get response from chat function with timeout protection
        try:
            response_text = chat_with_database(db_url, query)
        except MemoryError:
            logging.error("Memory error during chat processing")
            return Response(
                {"error": "Memory limit exceeded. Please try a simpler query."}, 
                status=status.HTTP_507_INSUFFICIENT_STORAGE
            )
        
        # Monitor memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024
        logging.info(f"Memory after processing: {final_memory:.2f} MB (used {final_memory - initial_memory:.2f} MB)")
        logging.info("Chat query processed successfully")
        
        return Response({
            "query": query,
            "response": response_text
        }, status=status.HTTP_200_OK)
        
    except MemoryError as e:
        logging.error(f"Memory error in chat endpoint: {str(e)}")
        return Response(
            {"error": "Memory limit exceeded. Please try a simpler query or restart the service."}, 
            status=status.HTTP_507_INSUFFICIENT_STORAGE
        )
    except Exception as e:
        logging.error(f"Error processing chat query: {str(e)}")
        return Response(
            {"error": f"Failed to process query: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def health_check(request):
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


class DashboardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = 'page_size'
    max_page_size = 100


class CustomerListView(ListAPIView):
    queryset = Customer.objects.all().order_by('-created_at')
    serializer_class = CustomerSerializer
    pagination_class = DashboardPagination


class CustomerDashboardView(RetrieveAPIView):
    queryset = Customer.objects.all()
    serializer_class = CustomerDashboardSerializer
    lookup_field = 'id'


class ContractListView(ListAPIView):
    serializer_class = ContractSerializer
    pagination_class = DashboardPagination
    
    def get_queryset(self):
        queryset = Contract.objects.all().select_related('customer').prefetch_related(
            'contract_products__product', 'billing_breakdowns'
        ).order_by('-created_at')
        
        customer_id = self.request.query_params.get('customer_id')
        if customer_id:
            queryset = queryset.filter(customer_id=customer_id)
            
        return queryset


class SupportTicketListView(ListAPIView):
    serializer_class = SupportTicketSerializer
    pagination_class = DashboardPagination
    
    def get_queryset(self):
        queryset = SupportTicket.objects.all().select_related('customer').prefetch_related(
            'history'
        ).order_by('-created_at')
        
        customer_id = self.request.query_params.get('customer_id')
        if customer_id:
            queryset = queryset.filter(customer_id=customer_id)
            
        status_filter = self.request.query_params.get('status')
        if status_filter:
            queryset = queryset.filter(status=status_filter)
            
        return queryset


@api_view(['GET'])
def dashboard_summary(request):
    """
    Dashboard summary endpoint providing overview statistics
    """
    try:
        customer_count = Customer.objects.count()
        active_contracts = Contract.objects.filter(
            billing_status__in=['paid', 'pending']
        ).count()
        open_tickets = SupportTicket.objects.filter(
            status__in=['open', 'in_progress']
        ).count()
        
        total_contract_value = Contract.objects.filter(
            billing_status__in=['paid', 'pending']
        ).aggregate(
            total_value=models.Sum('contract_value_usd')
        )['total_value'] or 0
        
        total_mrr = Contract.objects.filter(
            billing_status__in=['paid', 'pending']
        ).aggregate(
            total_mrr=models.Sum('client_mrr_usd')
        )['total_mrr'] or 0
        
        return Response({
            'customer_count': customer_count,
            'active_contracts': active_contracts,
            'open_tickets': open_tickets,
            'total_contract_value_usd': float(total_contract_value),
            'total_mrr_usd': float(total_mrr)
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logging.error(f"Error generating dashboard summary: {str(e)}")
        return Response(
            {"error": f"Failed to generate dashboard summary: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def customer_comprehensive_info(request, customer_id):
    """
    Comprehensive customer information endpoint returning:
    1. Customer name
    2. Products
    3. Contract value (USD)
    4. Client MRR (USD)
    5. Contract start date
    6. Contract end date
    7. Financial breakdown (billing cycle, billing breakdown, billing status)
    8. Complaint history (ticket number, ticket history)
    9. Contact information (contact name, email, phone number)
    """
    try:
        customer = Customer.objects.select_related('contact_info').prefetch_related(
            'contracts__contract_products__product',
            'contracts__billing_breakdowns',
            'support_tickets__history'
        ).get(id=customer_id)
        
        serializer = CustomerDashboardSerializer(customer)
        return Response(serializer.data, status=status.HTTP_200_OK)
        
    except Customer.DoesNotExist:
        return Response(
            {"error": "Customer not found"}, 
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        logging.error(f"Error retrieving customer comprehensive info: {str(e)}")
        return Response(
            {"error": f"Failed to retrieve customer information: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )