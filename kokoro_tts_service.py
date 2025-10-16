"""
Kokoro TTS Service using Kokoro-82M
Provides REST API for text-to-speech synthesis
"""

import base64
import io
import time
import wave
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from kokoro import KPipeline


# ============================================================================
# Pydantic Models for Request/Response
# ============================================================================

class VoiceInfo(BaseModel):
    id: str
    name: str
    gender: str
    accent: str
    description: str


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    available_voices: List[str]
    gpu_available: bool


class VoicesResponse(BaseModel):
    voices: List[VoiceInfo]


class SynthesizeRequest(BaseModel):
    text: str = Field(..., max_length=500)
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    format: str = Field(default="wav", pattern="^(wav|mp3)$")
    sample_rate: int = Field(default=24000)


class SynthesizeResponse(BaseModel):
    success: bool
    audio: str  # base64 encoded
    format: str
    duration: float
    sample_rate: int
    size_bytes: int
    metadata: dict


class BatchSegment(BaseModel):
    id: str
    text: str = Field(..., max_length=500)
    voice: str
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


class BatchRequest(BaseModel):
    segments: List[BatchSegment]
    format: str = Field(default="wav", pattern="^(wav|mp3)$")


class BatchSegmentResponse(BaseModel):
    id: str
    audio: str  # base64 encoded
    duration: float
    size_bytes: int


class BatchResponse(BaseModel):
    success: bool
    segments: List[BatchSegmentResponse]
    total_duration: float
    processing_time_ms: int


# ============================================================================
# Voice Configuration
# ============================================================================

AVAILABLE_VOICES = [
    VoiceInfo(
        id="af_bella",
        name="Bella",
        gender="female",
        accent="american",
        description="Warm, mature narrator voice"
    ),
    VoiceInfo(
        id="af_sarah",
        name="Sarah",
        gender="female",
        accent="american",
        description="Sophisticated, elegant"
    ),
    VoiceInfo(
        id="af_nicole",
        name="Nicole",
        gender="female",
        accent="irish",
        description="Playful, energetic"
    ),
    VoiceInfo(
        id="af_sky",
        name="Sky",
        gender="female",
        accent="american",
        description="Confident, knowing"
    ),
]

VOICE_IDS = {v.id for v in AVAILABLE_VOICES}
SAMPLE_RATE = 24000  # Kokoro uses 24kHz


# ============================================================================
# TTS Engine
# ============================================================================

class KokoroTTS:
    def __init__(self):
        """Initialize Kokoro TTS pipeline"""
        print("Loading Kokoro-82M model...")
        self.pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        self.loaded_voices = set()
        self.gpu_available = torch.cuda.is_available()
        print(f"Kokoro initialized. GPU available: {self.gpu_available}")

    def ensure_voice_loaded(self, voice_id: str):
        """Load voice if not already loaded"""
        if voice_id not in self.loaded_voices:
            print(f"Loading voice: {voice_id}")
            self.pipeline.load_voice(voice_id)
            self.loaded_voices.add(voice_id)

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> np.ndarray:
        """
        Synthesize speech from text

        Returns:
            numpy array of audio samples (float32, 24kHz)
        """
        if voice not in VOICE_IDS:
            raise ValueError(f"Invalid voice: {voice}")

        self.ensure_voice_loaded(voice)

        # Generate audio
        result_gen = self.pipeline(text, voice, speed=speed)
        result = list(result_gen)[0]

        # Convert torch tensor to numpy
        audio = result.audio.cpu().numpy()

        return audio

    def audio_to_wav(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio to WAV bytes"""
        # Normalize to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Kokoro TTS Service",
    description="Text-to-speech synthesis using Kokoro-82M",
    version="1.0"
)

# Initialize TTS engine at startup
tts_engine = None

@app.on_event("startup")
async def startup_event():
    global tts_engine
    tts_engine = KokoroTTS()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")

    return HealthResponse(
        status="healthy",
        model="kokoro-82m",
        version="1.0",
        available_voices=[v.id for v in AVAILABLE_VOICES],
        gpu_available=tts_engine.gpu_available
    )


@app.get("/voices", response_model=VoicesResponse)
async def list_voices():
    """List available voices"""
    return VoicesResponse(voices=AVAILABLE_VOICES)


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text"""
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")

    if request.voice not in VOICE_IDS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid voice ID: {request.voice}"
        )

    if request.format == "mp3":
        raise HTTPException(
            status_code=400,
            detail="MP3 format not yet implemented. Use 'wav'."
        )

    try:
        start_time = time.time()

        # Generate audio
        audio = tts_engine.synthesize(request.text, request.voice, request.speed)

        # Convert to WAV
        wav_bytes = tts_engine.audio_to_wav(audio)

        # Encode to base64
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        duration = len(audio) / SAMPLE_RATE
        processing_time = time.time() - start_time

        return SynthesizeResponse(
            success=True,
            audio=audio_b64,
            format="wav",
            duration=duration,
            sample_rate=SAMPLE_RATE,
            size_bytes=len(wav_bytes),
            metadata={
                "text": request.text,
                "voice": request.voice,
                "speed": request.speed,
                "processing_time_ms": int(processing_time * 1000)
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")


@app.post("/synthesize/batch", response_model=BatchResponse)
async def synthesize_batch(request: BatchRequest):
    """Batch synthesize multiple segments"""
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")

    if request.format == "mp3":
        raise HTTPException(
            status_code=400,
            detail="MP3 format not yet implemented. Use 'wav'."
        )

    try:
        start_time = time.time()
        results = []
        total_duration = 0.0

        for segment in request.segments:
            if segment.voice not in VOICE_IDS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid voice ID: {segment.voice}"
                )

            # Generate audio
            audio = tts_engine.synthesize(segment.text, segment.voice, segment.speed)

            # Convert to WAV
            wav_bytes = tts_engine.audio_to_wav(audio)

            # Encode to base64
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

            duration = len(audio) / SAMPLE_RATE
            total_duration += duration

            results.append(BatchSegmentResponse(
                id=segment.id,
                audio=audio_b64,
                duration=duration,
                size_bytes=len(wav_bytes)
            ))

        processing_time = int((time.time() - start_time) * 1000)

        return BatchResponse(
            success=True,
            segments=results,
            total_duration=total_duration,
            processing_time_ms=processing_time
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch synthesis error: {str(e)}")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
