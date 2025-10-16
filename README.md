# Kokoro TTS Service

**Status**: ✅ Production Ready
**Version**: 1.0
**Model**: Kokoro-82M

High-quality text-to-speech synthesis using the Kokoro-82M model with multiple character voices. Designed for narrative game dialogue generation with natural-sounding speech.

## Features

- **4 distinct voices**: American and Irish female voices with different personalities
- **Fast synthesis**: ~170ms per second of audio (realtime factor ~0.17x)
- **High quality**: 24kHz 16-bit mono WAV output
- **Batch processing**: Synthesize multiple segments in one request
- **Speed control**: Adjustable speaking rate (0.5-2.0x)
- **Base64 encoding**: Audio returned as base64 for easy integration

## Quick Start

### Prerequisites

- Python 3.11+ (required for kokoro dependencies)
- ~1GB RAM for model

### Installation

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Service

**Manual (for testing)**:
```bash
source venv/bin/activate
python kokoro_tts_service.py
```

**Production (systemd)**:
```bash
# Copy service file
sudo cp systemd/kokoro-tts.service /etc/systemd/system/

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable kokoro-tts
sudo systemctl start kokoro-tts

# Check status
sudo systemctl status kokoro-tts
```

## API Documentation

**Base URL**: `http://localhost:8001` or `http://192.168.1.95:8001`

### Health Check

```bash
curl http://localhost:8001/health
```

**Response**:
```json
{
  "status": "healthy",
  "model": "kokoro-82m",
  "version": "1.0",
  "available_voices": ["af_bella", "af_sarah", "af_nicole", "af_sky"],
  "gpu_available": false
}
```

### List Voices

```bash
curl http://localhost:8001/voices
```

**Response**:
```json
{
  "voices": [
    {
      "id": "af_bella",
      "name": "Bella",
      "gender": "female",
      "accent": "american",
      "description": "Warm, mature narrator voice"
    },
    {
      "id": "af_sarah",
      "name": "Sarah",
      "gender": "female",
      "accent": "american",
      "description": "Sophisticated, elegant"
    },
    {
      "id": "af_nicole",
      "name": "Nicole",
      "gender": "female",
      "accent": "irish",
      "description": "Playful, energetic"
    },
    {
      "id": "af_sky",
      "name": "Sky",
      "gender": "female",
      "accent": "american",
      "description": "Confident, knowing"
    }
  ]
}
```

### Synthesize Speech (Single)

```bash
curl -X POST http://localhost:8001/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "voice": "af_bella",
    "speed": 1.0,
    "format": "wav",
    "sample_rate": 24000
  }'
```

**Response**:
```json
{
  "success": true,
  "audio": "UklGRuABAABXQVZFZm10IBAAAA...",
  "format": "wav",
  "duration": 2.05,
  "sample_rate": 24000,
  "size_bytes": 98444,
  "metadata": {
    "text": "Hello, this is a test.",
    "voice": "af_bella",
    "speed": 1.0,
    "processing_time_ms": 345
  }
}
```

**Save audio to file**:
```bash
curl -X POST http://localhost:8001/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_bella"}' \
  | jq -r '.audio' | base64 -d > output.wav
```

### Batch Synthesis (Multiple Segments)

```bash
curl -X POST http://localhost:8001/synthesize/batch \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [
      {
        "id": "seg_001",
        "text": "The room falls silent.",
        "voice": "af_bella",
        "speed": 0.9
      },
      {
        "id": "seg_002",
        "text": "Hello, how can we help you?",
        "voice": "af_sarah",
        "speed": 0.95
      }
    ],
    "format": "wav"
  }'
```

**Response**:
```json
{
  "success": true,
  "segments": [
    {
      "id": "seg_001",
      "audio": "UklGRuABAABXQVZFZm10IBAAAA...",
      "duration": 1.8,
      "size_bytes": 28440
    },
    {
      "id": "seg_002",
      "audio": "UklGRuABAABXQVZFZm10IBAAAA...",
      "duration": 2.1,
      "size_bytes": 33120
    }
  ],
  "total_duration": 3.9,
  "processing_time_ms": 1270
}
```

## Voice Characteristics

### Recommended Character Assignments

Based on personality and tone:

| Voice | Speed | Best For | Personality |
|-------|-------|----------|-------------|
| **af_bella** | 0.9 | Narrator, mature characters | Warm, clear, authoritative |
| **af_sarah** | 0.95 | Sophisticated characters | Elegant, refined, intelligent |
| **af_nicole** | 1.05 | Young, energetic characters | Playful, lively, cheerful |
| **af_sky** | 1.0 | Confident leaders | Strong, knowing, direct |

### Speed Parameter Guidelines

- **0.5-0.8**: Slow, dramatic, or elderly characters
- **0.9-1.0**: Normal conversation pace (recommended)
- **1.1-1.3**: Fast-talking, excited, or young characters
- **1.4-2.0**: Comedic effect, time-skipping, or panic

## Performance

- **Synthesis speed**: ~170ms per second of audio
- **Example**: 2 seconds of audio = ~345ms processing
- **Batch overhead**: ~385ms per segment average
- **Memory usage**: ~1GB (model in RAM)
- **Concurrent requests**: Supported (FastAPI async)

## Architecture

```
┌─────────────────────┐
│  FastAPI Service    │  Port 8001
│  kokoro_tts         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Kokoro-82M Model   │  ~1GB in RAM
│  (via kokoro-onnx)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  24kHz WAV Output   │  Base64 encoded
│  16-bit Mono        │
└─────────────────────┘
```

## Integration Example (Python)

```python
import requests
import base64

TTS_URL = "http://192.168.1.95:8001"

def synthesize_dialogue(text: str, voice: str = "af_bella", speed: float = 1.0):
    """Synthesize a single line of dialogue"""
    response = requests.post(
        f"{TTS_URL}/synthesize",
        json={"text": text, "voice": voice, "speed": speed}
    )
    data = response.json()

    # Decode base64 audio
    audio_bytes = base64.b64decode(data["audio"])

    # Save to file
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)

    return data["duration"]

def synthesize_batch(segments: list):
    """Synthesize multiple segments at once"""
    response = requests.post(
        f"{TTS_URL}/synthesize/batch",
        json={"segments": segments, "format": "wav"}
    )
    data = response.json()

    # Save each segment
    for seg in data["segments"]:
        audio_bytes = base64.b64decode(seg["audio"])
        with open(f"{seg['id']}.wav", "wb") as f:
            f.write(audio_bytes)

    return data["total_duration"]

# Example usage
duration = synthesize_dialogue(
    text="Welcome to the adventure!",
    voice="af_bella",
    speed=0.95
)
print(f"Generated {duration:.2f} seconds of audio")
```

## Troubleshooting

### Service won't start
```bash
# Check logs
sudo journalctl -u kokoro-tts -n 50

# Verify Python version (must be 3.11+)
python --version

# Verify dependencies installed
source venv/bin/activate
pip list | grep kokoro
```

### Audio sounds distorted
- Ensure `speed` parameter is in range 0.5-2.0
- Check input text for proper punctuation
- Verify sample_rate is 24000 (default)

### Slow synthesis
- Check CPU usage: `htop`
- Verify only one instance is running
- Consider GPU acceleration (if available)

### Base64 decode error
- Ensure you're extracting `.audio` field from JSON response
- Use `jq -r '.audio'` to get raw string (not quoted)
- Verify complete response received (check content-length)

## Files

- `kokoro_tts_service.py` - Main FastAPI service
- `requirements.txt` - Python dependencies (kokoro-onnx 0.9.4)
- `systemd/kokoro-tts.service` - Systemd service file

## Integration

This service is part of the audio generation pipeline for narrative games. See the main documentation:

- [STINA AI Server Docs](https://github.com/onuse/stina-ai-server/tree/main/docs)
- [Audio Services Deployment Guide](https://github.com/onuse/stina-ai-server/blob/main/docs/AUDIO_SERVICES_DEPLOYMENT.md)

## Dependencies

- **kokoro-onnx**: 0.9.4 (TTS engine)
- **FastAPI**: Modern web framework
- **uvicorn**: ASGI server
- **pydantic**: Data validation

## License

MIT License - See repository for details.

## Related Services

- [Speaker Detection Service](https://github.com/onuse/speaker-detection-service) - Identifies speakers in narrative text
- [STINA AI Server](https://github.com/onuse/stina-ai-server) - Main LLM inference server
