# Step 1: Imports
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# Step 2: Read PDF
pdf_path = input("Enter PDF file name (e.g., resume.pdf): ").strip()

reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

# Step 3: Chunking
def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = chunk_text(text)
print("Chunks created:", len(chunks))

# Step 4: Embeddings(Local)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

#Step 5: Search 
def search(query, chunks, embeddings, top_k=5):

    query_embedding = model.encode([query])[0]

    similarities = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = [chunks[i] for i in top_indices]

    # Improve ranking for skill-related queries
    if "skill" in query.lower():
        results = sorted(
            results,
            key=lambda x: ("skill" in x.lower() or "experience" in x.lower()),
            reverse=True
        )

    return results

#Step 6: LLM
def generate_answer(query, results):
    context = "\n".join(results)

    prompt = f"""
You are an AI Resume Parser.

STRICT RULES:
- Extract ONLY from the context
- DO NOT assume anything
- DO NOT add explanations
- Return ALL relevant points (do not miss important details)
- Use bullet points

If question is about skills:
Return ALL skills found in the context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model='llama3',
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']

#Step 7:Main (Final chatbot loop)
while True:
    query = input("\nAsk something (or type 'exit'): ").strip()

    # Handle empty input
    if not query:
        print("⚠️ Please enter a question.")
        continue

    # Exit condition
    if query.lower() == "exit":
        break

    results = search(query, chunks, embeddings)

    answer = generate_answer(query, results)

    print("\nAnswer:\n")
    print(answer)
