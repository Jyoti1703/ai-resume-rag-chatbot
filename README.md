# AI Resume Analyzer (RAG Chatbot)

## Overview
Built a Retrieval-Augmented Generation (RAG) based chatbot that can analyze resumes and answer questions using Generative AI.

## Features
- Extracts data from PDF resumes
- Semantic search using embeddings
- Context-aware answers using LLM (Llama3 via Ollama)
- Local AI (no API cost)

## Tech Stack
- Python
- Sentence Transformers
- Ollama (Llama3)
- NumPy
- PyPDF

## How it works
1. Load PDF
2. Chunk text
3. Generate embeddings
4. Perform semantic search
5. Pass context to LLM
6. Generate answer

## Run the project
```bash
pip install -r requirements.txt
python read_pdf.py

## Example Questions
- What are the skills?
- Summarize the resume
- What experience does this person have?

## Future Improvements
- Integration with Azure OpenAI (GPT-4, embeddings) for scalable enterprise deployment
- Web UI using Streamlit for interactive chatbot experience
- Multi-document support (analyze multiple resumes at once)