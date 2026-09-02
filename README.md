# PDF Insight AI-Powered Document Q&A & Summarization

PDF Insight is a Retrieval-Augmented Generation (RAG) web app that lets you
upload any PDF and instantly **ask it questions in plain English** or
**generate a summary that scales to the document's own length** combining
fast local semantic search with Gemini for generation.

Upload a research paper, a contract, a report, or lecture notes, and get
grounded answers with the page number(s) they came from not hallucinated
guesses.

---
## 🎥 Demo [Watch the demo](assets/demo.mp4)

## Features

- **Semantic search over your document** using sentence embeddings, not
  brittle keyword matching
- **Natural-language question answering**, grounded in retrieved context,
  with source page numbers
- **Length-aware summarization** a short document gets a short summary,
  a long or dense one gets a fuller, multi-paragraph summary, instead of a
  fixed word cap
- **Lightweight** only a small embedding model runs locally; generation
  is offloaded to the Gemini API, so there's no multi-gigabyte model download
- Clean **Streamlit** interface with chat-style history

## How it works

```
                ┌────────────┐     ┌───────────────┐     ┌──────────────┐
   PDF Upload → │  Chunking  │ →   │   Embedding    │ →   │ FAISS Index  │
                │(pdfplumber)│     │(MiniLM sentence│     │ (vector store)│
                └────────────┘     │  transformer,  │     └──────────────┘
                                    │  local)        │             │
                                    └───────────────┘             │
                                                                   ▼
   User Question →  Embed question  →  Retrieve top-k chunks  →  Gemini
                                                                   generates
                                                                   grounded
                                                                   answer
```

1. **`pdf_processor.py`** extracts text page-by-page and splits it into
   overlapping chunks so context isn't lost at chunk boundaries.
2. **`embeddings.py`** embeds each chunk locally with `all-MiniLM-L6-v2` and
   indexes them in a FAISS similarity index this part needs no API key.
3. **`llm_engine.py`** embeds the user's question, retrieves the most
   relevant chunks, and sends them to Gemini (`gemini-3.5-flash-lite`) to generate
   a grounded answer or, for summarization, sends the whole document (with
   a map-reduce fallback for very large ones) and asks for a summary scaled
   to its length.
4. **`app.py`** ties it all together in a Streamlit UI, with the Gemini API
   key entered securely at runtime (never stored or committed).

## Tech stack

| Layer              | Tool                                          |
|---------------------|------------------------------------------------|
| UI                  | Streamlit                                       |
| PDF parsing         | pdfplumber                                      |
| Embeddings (local)  | sentence-transformers (`all-MiniLM-L6-v2`)      |
| Vector search       | FAISS                                           |
| Answer generation   | Google Gemini API (`gemini-3.5-flash-lite`)          |
| Summarization       | Google Gemini API (`gemini-3.5-flash-lite`)          |

## Getting started

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/<your-username>/pdf-insight.git
cd pdf-insight
python -m venv venv
source venv/Scripts/activate      # MacBook: venv\bin\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a free Gemini API key

Grab one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
(free tier available). Two ways to use it:

- **Quick/one-off:** paste it into the app's sidebar at runtime. It's only
  kept in that session's memory never written to disk or committed.
- **For your own dev machine:** copy `.env.example` to `.env` and put your
  key in there instead, so you don't have to re-type it every run:
  ```bash
  cp .env.example .env
  # then edit .env and set GEMINI_API_KEY=your_actual_key
  ```
  `.env` is already listed in `.gitignore`, so `git status` will never show
  it as a tracked or staged file it stays local and is never pushed, even
  if the repo is public.

### 4. Run the app

```bash
streamlit run app.py
```

The first run will download the small local embedding model
(~80 MB) — this only happens once.

### 4. Run the tests

```bash
pytest tests/
```

## Project structure

```
pdf-insight/
├── app.py                     # Streamlit UI
├── src/
│   ├── pdf_processor.py       # PDF text extraction + chunking
│   ├── embeddings.py          # Local embedding model + FAISS vector store
│   └── llm_engine.py          # Gemini-powered QA and summarization
├── tests/
│   └── test_pdf_processor.py  # Unit tests for chunking logic
├── requirements.txt
└── README.md
```

## 🗺️ Future improvements

- [ ] Support multi-document sessions with cross-document search
- [ ] Add persistent vector storage so re-uploading isn't required
- [ ] Docker support for one-command deployment
- [ ] Highlight the exact answer span inside the source PDF page
- [ ] Optional local-only mode (no API key) using a small local LLM

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

---

Built by **Mahnoor** — BS Artificial Intelligence, Government College
University (GCU) Lahore.
