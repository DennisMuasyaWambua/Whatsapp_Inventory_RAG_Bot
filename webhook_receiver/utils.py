import logging, json, re, requests, os
from django.http import JsonResponse
from webhook_receiver.chat import chat_with_database
from django.conf import settings

def log_http_response(response):
    """Log detailed HTTP response information for debugging."""
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Content-type: {response.headers.get('content-type')}")
    
    # Try to parse and log JSON response for better debugging
    try:
        json_response = response.json()
        logging.info(f"Response JSON: {json.dumps(json_response, indent=2)}")
    except ValueError:
        # If not JSON, log as text
        logging.info(f"Response Body: {response.text}")
    
    # Log all headers for debugging
    logging.info("Response Headers:")
    for header, value in response.headers.items():
        logging.info(f"  {header}: {value}")
        
    # If error status code, provide more detailed information
    if response.status_code >= 400:
        logging.error(f"Error response received: HTTP {response.status_code}")
        if response.status_code == 401:
            logging.error("AUTHENTICATION ERROR: Your access token may be invalid or expired")
            logging.error("Please check your ACCESS_TOKEN environment variable or settings")


def get_text_message_input(recipient, text):
    return json.dumps(
        {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": recipient,
            "type": "text",
            "text": {"preview_url": False, "body": text},
        }
    )


def generate_response(response):
    # Return text in uppercase
    return response.upper()


def send_message(data):
    
    
    # Get access token from either environment variable or Django settings
    access_token = os.environ.get('ACCESS_TOKEN') or settings.WHATSAPP_ACCESS_TOKEN
    
    if not access_token:
        logging.error("ACCESS_TOKEN is not set in environment or settings")
        return {"status": "error", "message": "ACCESS_TOKEN is not configured"}
    
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # Log the API credentials being used (exclude sensitive parts of the token)
    token_preview = access_token[:5] + '...' if access_token else 'NOT SET'
    
    logging.info(f"Using API version: {settings.WHATSAPP_API_VERSION}")
    logging.info(f"Using phone number ID: {os.environ.get('PHONE_NUMBER_ID') or settings.WHATSAPP_PHONE_NUMBER_ID}")
    logging.info(f"Using access token: {token_preview}")
    
    # Get API version from either environment variable or Django settings
    api_version = os.environ.get('VERSION') or settings.WHATSAPP_API_VERSION
    phone_id = os.environ.get('PHONE_NUMBER_ID') or settings.WHATSAPP_PHONE_NUMBER_ID
    
    # Log the complete URL for debugging
    url = f"https://graph.facebook.com/{api_version}/{phone_id}/messages"
    logging.info(f"Full API URL being used: {url}")

    # Check if required credentials are available
    if not access_token:
        error_msg = "Missing WhatsApp API access token. Please configure ACCESS_TOKEN environment variable or WHATSAPP_ACCESS_TOKEN in settings."
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}
        
    if not phone_id:
        error_msg = "Missing WhatsApp phone number ID. Please configure PHONE_NUMBER_ID environment variable or WHATSAPP_PHONE_NUMBER_ID in settings."
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}
        
    if not api_version:
        error_msg = "Missing WhatsApp API version. Please configure VERSION environment variable or WHATSAPP_API_VERSION in settings."
        logging.error(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        logging.info(f"Sending message to WhatsApp API: {url}")
        
        response = requests.post(
            url, data=data, headers=headers, timeout=10
        )  # 10 seconds timeout as an example
        
        # Log the full response for debugging
        log_http_response(response)
        
        response.raise_for_status()  # Raises an HTTPError if the HTTP requests returned an unsuccessful status code
    except requests.Timeout:
        logging.error("Timeout occurred while sending message")
        return {"status": "error", "message": "Request timed out"}
    except (
        requests.RequestException
    ) as e:  # This will catch any general requests exception
        if isinstance(e, requests.HTTPError):
            try:
                error_details = e.response.json() if e.response.text else {}
                error_message = error_details.get('error', {}).get('message', 'No error details available')
                error_code = error_details.get('error', {}).get('code', 'unknown')
                error_type = error_details.get('error', {}).get('type', 'unknown')
                
                if e.response.status_code == 401:
                    logging.error(f"Authentication error (401): {error_message}")
                    logging.error(f"This is likely due to an invalid or expired ACCESS_TOKEN")
                    logging.error(f"Current token: {access_token[:10]}... (first 10 chars)")
                    logging.error(f"Make sure you've sourced setup_env.sh and have a valid token")
                    return {"status": "error", "message": f"Authentication failed: {error_message}"}
                
                elif e.response.status_code == 400:
                    logging.error(f"Bad Request error (400): {error_message}")
                    logging.error(f"Error code: {error_code}, Error type: {error_type}")
                    logging.error(f"Request data: {data}")
                    logging.error(f"This may be due to invalid message format or content")
                    
                    # Log specific common errors
                    if "message recipient not found" in error_message.lower():
                        logging.error(f"The recipient number may be invalid or not registered on WhatsApp")
                    elif "unsupported message type" in error_message.lower():
                        logging.error(f"The message type is not supported")
                    return {"status": "error", "message": f"Bad request: {error_message}"}
                
                else:
                    logging.error(f"HTTP error {e.response.status_code}: {error_message}")
                    return {"status": "error", "message": f"API error: {error_message}"}
            except ValueError:
                logging.error(f"Request failed due to: {e}")
                logging.error(f"Could not parse error response: {e.response.text if hasattr(e, 'response') else 'No response'}")
                return {"status": "error", "message": f"Failed with status {e.response.status_code if hasattr(e, 'response') else 'unknown'}"}
        else:
            logging.error(f"Request failed due to: {e}")
            return {"status": "error", "message": "Failed to send message"}
    else:
        # Process the response as normal
        return response


def process_text_for_whatsapp(text):
    # Remove brackets
    pattern = r"\【.*?\】"
    # Substitute the pattern with an empty string
    text = re.sub(pattern, "", text).strip()

    # Pattern to find double asterisks including the word(s) in between
    pattern = r"\*\*(.*?)\*\*"

    # Replacement pattern with single asterisks
    replacement = r"*\1*"

    # Substitute occurrences of the pattern with the replacement
    whatsapp_style_text = re.sub(pattern, replacement, text)

    return whatsapp_style_text


def process_whatsapp_message(body):
    try:
        # Extract message data with better error handling
        try:
            contacts = body.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {}).get("contacts", [])
            if not contacts:
                logging.error(f"No contacts found in webhook payload: {json.dumps(body)}")
                return
                
            wa_id = contacts[0].get("wa_id")
            name = contacts[0].get("profile", {}).get("name", "Unknown")
            
            messages = body.get("entry", [{}])[0].get("changes", [{}])[0].get("value", {}).get("messages", [])
            if not messages:
                logging.error(f"No messages found in webhook payload: {json.dumps(body)}")
                return
                
            message = messages[0]
            message_type = message.get("type")
            message_id = message.get("id", "unknown_message_id")
            
            # Handle different message types
            if message_type == "text":
                message_body = message.get("text", {}).get("body", "")
            else:
                logging.info(f"Received non-text message of type: {message_type}")
                message_body = f"[Received {message_type} message]"
                
        except (KeyError, IndexError) as e:
            logging.error(f"Error extracting message data: {e}")
            logging.error(f"Webhook payload: {json.dumps(body)}")
            return
        
        logging.info(f"Processing message from {name} ({wa_id}): {message_body}")
        
        # Process the user's query through the RAG system
        from webhook_receiver.chat import chat_with_database
        from django.conf import settings
        
        # Get database response from RAG
        logging.info(f"Querying RAG system with: {message_body}")
        
        try:
            # Pass the user's message to the RAG system
            ai_response = chat_with_database(settings.DB_URL, query=message_body)
            
            logging.info(f"RAG response received: {ai_response}")
            
            # Process the RAG response for WhatsApp formatting if needed
            formatted_response = process_text_for_whatsapp(ai_response)
            
        except Exception as e:
            logging.error(f"Error getting RAG response: {str(e)}", exc_info=True)
            formatted_response = "Sorry, I encountered an error while searching the database. Please try again later."
        
        # Create the WhatsApp message with the AI response
        # Use the wa_id from the incoming message as the recipient (reply to sender)
        data = get_text_message_input(wa_id, formatted_response)
        
        # Log what we're about to send
        logging.info(f"Sending response to {wa_id} (message ID: {message_id}): {formatted_response[:100]}...")
        
        # Send the message
        result = send_message(data)
        
        # Log the result
        if isinstance(result, dict) and result.get("status") == "error":
            logging.error(f"Failed to send message: {result.get('message')}")
        else:
            logging.info(f"Message sent successfully to {wa_id}")
            
    except Exception as e:
        logging.error(f"Error in process_whatsapp_message: {str(e)}", exc_info=True)
        # Don't re-raise to prevent webhook failure response


def is_valid_whatsapp_message(body):
    """
    Check if the incoming webhook event has a valid WhatsApp message structure.
    
    Returns:
        bool: True if this is a valid WhatsApp message event, False otherwise
    """
    try:
        # Log the structure for debugging
        logging.debug(f"Validating webhook structure: {json.dumps(body)}")
        
        # Safely check the structure without risking KeyError or IndexError
        if body.get("object") != "whatsapp_business_account":
            logging.warning(f"Invalid object type: {body.get('object')}")
            return False
            
        if not body.get("entry") or not isinstance(body.get("entry"), list) or len(body.get("entry", [])) == 0:
            logging.warning("Missing or empty 'entry' array")
            return False
            
        entry = body.get("entry", [{}])[0]
        if not entry.get("changes") or not isinstance(entry.get("changes"), list) or len(entry.get("changes", [])) == 0:
            logging.warning("Missing or empty 'changes' array")
            return False
            
        change = entry.get("changes", [{}])[0]
        if not change.get("value"):
            logging.warning("Missing 'value' object in change")
            return False
            
        value = change.get("value", {})
        
        # Verify this is a WhatsApp message
        if value.get("messaging_product") != "whatsapp":
            logging.warning(f"Invalid messaging_product: {value.get('messaging_product')}")
            return False
            
        # Check if this contains messages (could be a status update instead)
        if not value.get("messages") or not isinstance(value.get("messages"), list) or len(value.get("messages", [])) == 0:
            # This might be a status update, not a message
            logging.info("No messages found - this may be a status update")
            return False
            
        # Finally check for a valid message
        message = value.get("messages", [{}])[0]
        if not message.get("id") or not message.get("from"):
            logging.warning("Message missing required fields (id, from)")
            return False
            
        # Check for contacts
        if not value.get("contacts") or not isinstance(value.get("contacts"), list) or len(value.get("contacts", [])) == 0:
            logging.warning("Missing or empty 'contacts' array")
            return False
            
        # All checks passed
        return True
        
    except Exception as e:
        logging.error(f"Error validating WhatsApp message: {str(e)}")
        return False

def handle_message(request=None):
    """
    Handle incoming webhook events from the WhatsApp API.

    This function processes incoming WhatsApp messages and other events,
    such as delivery statuses. If the event is a valid message, it gets
    processed. If the incoming payload is not a recognized WhatsApp event,
    an error is returned.

    Every message send will trigger 4 HTTP requests to your webhook: message, sent, delivered, read.

    Returns:
        response: A tuple containing a JSON response and an HTTP status code.
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
        logging.info(f"Request body: {body}")
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON")
        return {"status": "error", "message": "Invalid JSON provided"}, 400

    # Check if it's a WhatsApp status update
    if (
        body.get("entry", [{}])[0]
        .get("changes", [{}])[0]
        .get("value", {})
        .get("statuses")
    ):
        logging.info("Received a WhatsApp status update.")
        return {"status": "ok"}, 200

    try:
        if is_valid_whatsapp_message(body):
            process_whatsapp_message(body)
            return {"status": "ok"}, 200
        else:
            # if the request is not a WhatsApp API event, return an error
            logging.warning("Not a WhatsApp API event")
            return {"status": "error", "message": "Not a WhatsApp API event"}, 400
    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        return {"status": "error", "message": f"Error processing message: {str(e)}"}, 500


# Required webhook verification for WhatsApp
def verify(request=None):
    # Parse params from the webhook verification request
    mode = request.GET.get("hub.mode")
    token = request.GET.get("hub.verify_token")
    challenge = request.GET.get("hub.challenge")
    # mode = "subscribe"
    # token = settings.WHATSAPP_VERIFY_TOKEN
    # challenge = "test123"
    
    # For debugging
    logging.info(f"Webhook verification request received")
    logging.info(f"Mode: {mode}")
    logging.info(f"Token received: {token}")
    logging.info(f"Challenge: {challenge}")
    
    # Get the verification token from settings

    verify_token = settings.WHATSAPP_VERIFY_TOKEN
    
    logging.info(f"Expected token: {verify_token or 'NOT SET'}")
    
    # For direct browser access without parameters, return a helpful message
    if not mode and not token and not challenge:
        logging.info("Direct access to webhook URL detected")
        return {
            "status": "ok", 
            "message": "WhatsApp Webhook endpoint. To verify this webhook, please send a GET request with hub.mode, hub.verify_token, and hub.challenge parameters."
        }, 200
    
    # For testing/development, accept any token if verify_token is not set
    if not verify_token:
        logging.warning("WHATSAPP_VERIFY_TOKEN not set, accepting any token for verification during development")
        if mode == "subscribe" and token and challenge:
            return challenge, 200
    
    # Check if all required parameters were sent
    if not mode or not token or not challenge:
        logging.warning(f"Missing required parameters: mode={mode}, token={token}, challenge={challenge}")
        return {
            "status": "error", 
            "message": "Missing required parameters. This webhook requires hub.mode, hub.verify_token, and hub.challenge parameters."
        }, 400
    
    # Check if the mode and token sent are correct
    if mode == "subscribe" and (token == verify_token or verify_token is None):
        # Respond with 200 OK and challenge token from the request
        logging.info("WEBHOOK_VERIFIED")
        return challenge, 200
    else:
        # Responds with '403 Forbidden' if verify tokens do not match
        logging.info("VERIFICATION_FAILED")
        logging.info(f"Token mismatch: received '{token}', expected '{verify_token}'")
        return {"status": "error", "message": "Verification failed"}, 403