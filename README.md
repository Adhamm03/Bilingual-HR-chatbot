# Bilingual HR Chatbot

A bilingual (Arabic/English) legal assistant for Sharjah's HR law. Ask questions in either language and get answers grounded in the actual legal text, with conversation history, voice I/O, and user authentication.

## How It Works

1. The Arabic legal document is chunked by articles and tables
2. Chunks are indexed using **FAISS** (semantic) + **BM25** (keyword) search
3. Both results are merged via **Reciprocal Rank Fusion (RRF)**
4. **GPT-4o-mini** generates the final answer from the retrieved chunks
5. English queries are auto-translated to Arabic for better retrieval
6. Three suggested follow-up questions are generated with each answer

## Setup

```bash
git clone https://github.com/Adhamm03/Bilingual-HR-chatbot.git
cd Bilingual-HR-chatbot
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-your-api-key-here
```

## Run

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 in your browser.

## Run with Docker

1. Create a `.env` file in the project root directory and add your OpenAI API key:
   ```env
   OPENAI_API_KEY=sk-your-api-key-here
   ```

2. Build the Docker containers:
   ```bash
   docker-compose build
   ```

3. Start the application:
   ```bash
   docker-compose up
   ```

4. Open [http://localhost:8000](http://localhost:8000) in your browser.

5. Stop the application:
   ```bash
   docker-compose down
   ```

## Project Structure

```
app.py                      # FastAPI server
rag_engine.py               # RAG engine (chunking, retrieval, generation)
arabic_text_and_tables.txt  # Source legal document
static/
  chat.html                 # Chat UI
requirements.txt            # Python dependencies
Dockerfile
docker-compose.yml
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Chat UI |
| POST | `/ask` | Ask a question (`{"question": "..."}`) |
| POST | `/transcribe` | Speech-to-text (upload audio file) |
| POST | `/tts` | Text-to-speech (`{"text": "..."}`) |
| GET | `/ping` | Health check |
| GET | `/stats` | System stats (model info, retrieval config) |

## Tech Stack

- **FastAPI** — web server
- **Sentence Transformers** (`multilingual-e5-large`) — embeddings
- **FAISS** — vector search
- **BM25** — keyword search
- **OpenAI GPT-4o-mini** — translation + answer generation
- **OpenAI Whisper** — speech-to-text
- **OpenAI TTS** — text-to-speech
- **SQLite** — user accounts and conversation history
