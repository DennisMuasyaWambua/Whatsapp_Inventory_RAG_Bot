# WhatsApp Inventory Chat Bot

A Django application that processes WhatsApp messages and responds based on database queries.

## Setup Instructions

### 1. Configure Environment Variables

Create a `.env` file in the project root with the following variables:

```
# WhatsApp API Credentials
WHATSAPP_VERIFY_TOKEN=your_verification_token
VERSION=v17.0
PHONE_NUMBER_ID=your_whatsapp_phone_number_id
ACCESS_TOKEN=your_whatsapp_access_token

# Database Configuration (optional)
DB_URL=sqlite:///db.sqlite3
```

#### Required Environment Variables

- `WHATSAPP_VERIFY_TOKEN`: A custom token used for webhook verification (can be any string you choose)
- `VERSION`: WhatsApp API version (recommend using v17.0)
- `PHONE_NUMBER_ID`: Your WhatsApp Phone Number ID from the Meta Developer Dashboard
- `ACCESS_TOKEN`: Your WhatsApp API Permanent Access Token from the Meta Developer Dashboard

#### How to Get WhatsApp API Credentials

1. Go to [Meta Developers](https://developers.facebook.com/)
2. Create or select an app 
3. Add the WhatsApp product
4. Get your Phone Number ID and API Access Token from the WhatsApp setup page

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Database Migrations

```bash
python manage.py migrate
```

### 4. Start the Server

```bash
python manage.py runserver
```

## Exposing Your Server for Webhook Testing

To test with actual WhatsApp messages, you need to expose your local server to the internet. You can use:

- [ngrok](https://ngrok.com/) - `ngrok http 8000`
- [localtunnel](https://localtunnel.github.io/www/) - `lt --port 8000`

Use the generated URL as your WhatsApp webhook URL.

## Setting Up WhatsApp Webhook

1. In the Meta Developer Dashboard, go to WhatsApp > Configuration
2. Add a webhook URL (your exposed URL + `/api/webhook`)
3. Use your `WHATSAPP_VERIFY_TOKEN` as the verification token
4. Subscribe to the `messages` webhook field

### Testing the Webhook Manually

To manually test the webhook verification, visit your webhook URL with the following query parameters:

```
https://your-domain.com/api/webhook?hub.mode=subscribe&hub.verify_token=your_token&hub.challenge=challenge_string
```

Replace:
- `your-domain.com` with your actual domain
- `your_token` with your WHATSAPP_VERIFY_TOKEN value
- `challenge_string` with any random string

If verification is successful, the server will respond with the challenge string. If there are any issues, you'll receive a JSON response with error details.

## Troubleshooting

### "401 Unauthorized" Error

If you see this error when sending messages:
```
ERROR:root:Request failed due to: 401 Client Error: Unauthorized for url: https://graph.facebook.com/v17.0/XXXXX/messages
```

The problem is with your WhatsApp API credentials. Follow these steps:

1. **Get a valid access token**
   - Go to the Meta Developer Dashboard > WhatsApp > Configuration
   - Generate a permanent access token (recommended) or use a temporary one for testing
   - The token must have the "whatsapp_business_messaging" permission

2. **Set environment variables**
   - Edit the `setup_env.sh` script and add your access token
   - Run `source setup_env.sh` to load the variables
   - Restart your Django server

3. **Check API version compatibility**
   - We've set the default API version to v17.0 as it's more stable
   - Make sure your Meta Developer app is configured for the same version

4. **Verify your phone number ID**
   - This should match the phone number ID in your Meta Developer Dashboard
   - Currently set to: `558311380709590` (based on your error message)

For security reasons, never commit your access token to the repository.