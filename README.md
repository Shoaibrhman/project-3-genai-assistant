# Project 3: GenAI Domain Assistant — Company Knowledge RAG System

A complete, production-ready Retrieval Augmented Generation (RAG) assistant built over 4 weeks (Weeks 9-12) of the Applied AI course, culminating in a deployed Streamlit web app.

## Overview
This assistant answers questions about company policies (vacation, remote work, parental leave, benefits, IT) by retrieving relevant information from real documents and generating grounded, accurate answers — instead of relying on generic AI knowledge.

## What it does
- Chat interface with persistent conversation history (Streamlit session state)
- Semantic search powered by Google Gemini embeddings (`gemini-embedding-001`) and a ChromaDB vector database
- RAG pipeline: retrieves relevant document chunks, then generates answers using `gemini-3.6-flash`, grounded strictly in the retrieved context
- Sidebar with app info, live stats (documents indexed, messages sent), and a clear-chat button
- Friendly welcome message and loading spinner during search
- Error handling for missing database or API issues

## Tech stack
- Google Gemini API (`google-genai`) — embeddings and generation
- ChromaDB — persistent vector database
- Streamlit — web UI and chat interface
- LangChain — document loading and chunking (used in earlier weeks to build the indexed database)

## Project journey
- **Week 9:** Basic conversational chatbot using the Gemini API
- **Week 10:** Keyword-based RAG system — document loading, chunking, simple retrieval
- **Week 11:** Upgraded to semantic search — embeddings, cosine similarity, ChromaDB vector search (finds "vacation" from "PTO" or "time off," even with no shared keywords)
- **Week 12 (this repo):** Production Streamlit app tying everything together into a deployable, portfolio-ready assistant

## Files
- `app.py` — main Streamlit application
- `company_docs/` — sample company policy documents (HR, benefits, IT)

## Note
Requires a `.env` file with `GEMINI_API_KEY=your_key_here` and a populated `chroma_db/` (see Week 11 for indexing steps).
The `chroma_db/` folder (vector database) and `.env` (API key) are excluded via `.gitignore` and generated/set up locally, not included in this repo.

## Running locally
