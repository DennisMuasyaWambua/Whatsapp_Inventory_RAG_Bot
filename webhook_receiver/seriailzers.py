# webhook_receiver/serializers.py
from rest_framework import serializers
from .models import (
    WhatsAppMessage, Customer, CustomerData, Product, Contract, ContractProduct,
    BillingBreakdown, SupportTicket, TicketHistory, ContactInformation
)

class WhatsAppMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = WhatsAppMessage
        fields = ['id', 'sender', 'recipient', 'message_type', 'content', 'media_id', 'timestamp', 'created_at']
        read_only_fields = ['created_at']

class SendMessageSerializer(serializers.Serializer):
    recipient = serializers.CharField(max_length=50)
    message = serializers.CharField()

class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ['id', 'name', 'description', 'created_at']

class ContractProductSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)
    
    class Meta:
        model = ContractProduct
        fields = ['id', 'product', 'quantity', 'unit_price_usd']

class BillingBreakdownSerializer(serializers.ModelSerializer):
    class Meta:
        model = BillingBreakdown
        fields = ['id', 'description', 'amount_usd', 'percentage']

class ContactInformationSerializer(serializers.ModelSerializer):
    class Meta:
        model = ContactInformation
        fields = ['id', 'contact_name', 'email', 'phone_number']

class TicketHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = TicketHistory
        fields = ['id', 'action', 'description', 'created_by', 'created_at']

class SupportTicketSerializer(serializers.ModelSerializer):
    history = TicketHistorySerializer(many=True, read_only=True)
    
    class Meta:
        model = SupportTicket
        fields = [
            'id', 'ticket_number', 'subject', 'description', 'status', 
            'priority', 'created_at', 'updated_at', 'resolved_at', 'history'
        ]

class ContractSerializer(serializers.ModelSerializer):
    contract_products = ContractProductSerializer(many=True, read_only=True)
    billing_breakdowns = BillingBreakdownSerializer(many=True, read_only=True)
    
    class Meta:
        model = Contract
        fields = [
            'id', 'contract_value_usd', 'client_mrr_usd', 'start_date', 
            'end_date', 'billing_cycle', 'billing_status', 'created_at', 
            'updated_at', 'contract_products', 'billing_breakdowns'
        ]

class CustomerDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomerData
        fields = ['id', 'customer_name', 'products', 'contract_value_usd', 'client_mrr_usd', 
                  'contract_start_date', 'contract_end_date', 'billing_cycle', 'billing_breakdown',
                  'billing_status', 'ticket_number', 'ticket_history', 'contact_name', 
                  'contact_email', 'contact_phone']

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = ['id', 'customer_name', 'created_at', 'updated_at']

class CustomerDashboardSerializer(serializers.ModelSerializer):
    contracts = ContractSerializer(many=True, read_only=True)
    support_tickets = SupportTicketSerializer(many=True, read_only=True)
    contact_info = ContactInformationSerializer(read_only=True)
    
    class Meta:
        model = Customer
        fields = [
            'id', 'customer_name', 'created_at', 'updated_at',
            'contracts', 'support_tickets', 'contact_info'
        ]