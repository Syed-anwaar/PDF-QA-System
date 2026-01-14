import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------------
# STEP 1: Load API Key
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# STEP 2: App UI
# ----------------------------
st.set_page_config(page_title="PDF Q&A App", layout="wide")
st.title("📄 PDF Question Answering App")
st.write("Upload a PDF and ask questions based on its content.")

# ----------------------------
# STEP 3: Upload PDF
# ----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    # ----------------------------
    # STEP 4: Chunk PDF
    # ----------------------------
    def chunk_text(text, size=300, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - overlap):
            chunks.append(" ".join(words[i:i+size]))
        return chunks

    chunks = chunk_text(text)

    # ----------------------------
    # STEP 5: Embeddings + FAISS
    # ----------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # ----------------------------
    # STEP 6: Ask Question
    # ----------------------------
    question = st.text_input("Ask a question")

    if question:
        query_embedding = model.encode([question])
        _, indices = index.search(query_embedding, k=3)

        context = " ".join([chunks[i] for i in indices[0]])

        # ----------------------------
        # STEP 7: AI Answer
        # ----------------------------
        prompt = f"""
        Answer the question using ONLY the context below.
        Give 2–3 bullet points.

        Context:
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
