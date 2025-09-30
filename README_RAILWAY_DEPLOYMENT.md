# Railway Deployment with Ollama and Llama 3.2:1b

## Quick Setup

1. **Connect to Railway:**
   ```bash
   railway login
   railway link
   ```

2. **Deploy:**
   ```bash
   railway up
   ```

## Configuration Files Created

- `Dockerfile` - Multi-stage build with Ollama + your app
- `railway.toml` - Railway service configuration
- `setup_ollama.sh` - Ollama setup script
- `health_check.py` - Health monitoring endpoint

## Environment Variables

Set these in Railway dashboard:

```
OLLAMA_HOST=0.0.0.0:11434
OLLAMA_ORIGINS=*
DB_URL=your_database_url
PORT=8000
```

## Resource Requirements

- **CPU:** 2 vCPUs minimum
- **RAM:** 4GB minimum (for Llama 3.2:1b)
- **Storage:** 2GB+ (for model files)

## Health Check

Railway will monitor: `/health`
- Checks Ollama service
- Verifies model availability
- Lists loaded models

## Troubleshooting

1. **Ollama not starting:**
   - Check Railway logs
   - Verify resources allocated
   - Model download may take 5-10 minutes

2. **Model not found:**
   - Check `/models` endpoint
   - Verify `llama3.2:1b` was downloaded
   - Try `llama3.2:1b-instruct` as fallback

3. **Memory issues:**
   - Increase Railway plan
   - Reduce batch sizes in `chat.py`
   - Monitor `/health` endpoint

## Model Alternatives

If `llama3.2:1b` fails, the code tries:
- `llama3.2:1b-instruct`
- Falls back to structured responses without LLM