
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
from rag_engine import RAGEngine

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"
    model: str = "tts-1"

class AskRequest(BaseModel):
    question: str


app = FastAPI(
    title='Sharjah Rag',
    version='0.1.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

rag = None

@app.on_event("startup")
async def startup_event():
    global rag
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    print("Loading RAG engine...")
    rag = RAGEngine(
        document_path="arabic_text_and_tables.txt",
        openai_api_key=openai_api_key,
    )
    print(f"RAG engine loaded successfully! Total chunks: {len(rag.chunks)}")


@app.get("/")
def root():
    return FileResponse("static/chat.html")


@app.get("/api/info")
def api_info():
    return {
        "message": "Sharjah RAG API",
        "version": "0.1.0",
        "endpoints": {
            "POST /ask": "Ask a question",
            "GET /ping": "Health check",
            "GET /stats": "Get system stats"
        }
    }


@app.get("/ping")
def ping():
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return {
        "status": "healthy",
        "chunks_loaded": len(rag.chunks)
    }


@app.get("/stats")
def stats():
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return {
        "total_chunks": len(rag.chunks),
        "embedding_model": "intfloat/multilingual-e5-large",
        "llm_model": "gpt-4o-mini",
        "retrieval_methods": ["FAISS", "BM25"]
    }


@app.post('/ask')
def ask(payload: AskRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")

    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer, context = rag.answer_question(payload.question, debug=True, return_context=True)
        lang = rag.detect_language(payload.question)
        follow_up_questions = rag.generate_followup_questions(payload.question, answer, context, lang)
        return {
            "answer": answer,
            "follow_up_questions": follow_up_questions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/tts")
async def text_to_speech(payload: TTSRequest):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.audio.speech.create(
            model=payload.model,
            voice=payload.voice,
            input=payload.text
        )
        audio_data = response.content
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/mpeg",
            headers={"Content-Length": str(len(audio_data))}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        audio_bytes = await audio.read()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = audio.filename or "recording.webm"

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            response_format="verbose_json",
            prompt="""This assistant supports both English and Arabic. The user may speak in either language
            your answer must be in the same language as the users's ."""
        )

        print(f"Detected Language: {result.language}")
        print(f"Text: {result.text}")

        return {"language": result.language, "text": result.text}

    except Exception as e:
        print(f"Transcription error: {e}")
        return {"error": str(e)}
