import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# load api key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("PDF Question Answering System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # split text into chunks
    def chunk_text(text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunks.append(" ".join(words[i:i + chunk_size]))
        return chunks

    chunks = chunk_text(text)

    # create embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # faiss index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    question = st.text_input("Ask a question from the PDF")

    if question:
        query_embedding = model.encode([question]).astype("float32")
        _, indices = index.search(query_embedding, k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        prompt = f"""
Answer the question using the text below.
Give short bullet points.

Text:
{context}

Question:
{question}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        st.subheader("Answer")
        st.write(response.choices[0].message.content)
