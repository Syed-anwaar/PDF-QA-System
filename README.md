# PDF Question Answering System

A Python-based system that allows users to ask questions and receive concise bullet-point answers directly from a PDF document.

## Overview

This project extracts text from a PDF file and uses semantic search to return relevant answers strictly from the document content. If the answer is not present in the PDF, the system avoids generating incorrect responses.

## Technologies Used

- Python
- pypdf (PyPDF2 successor)
- SentenceTransformers
- FAISS
- Pandas
- NumPy

## Features

- Question answering from PDF content
- Short, 2–3 bullet-point answers
- Avoids out-of-context responses
- Works in Jupyter Notebook
- Supports any text-based PDF

## How It Works

1. Extract text from the PDF
2. Split text into manageable chunks
3. Generate semantic embeddings
4. Store embeddings using FAISS
5. Match user questions with relevant content
6. Generate concise bullet-point answers

## Usage

1. Open the Jupyter Notebook
2. Place your PDF inside the `data/` folder
3. Update the PDF path if required
4. Run all cells
5. Start asking questions

Type `exit` to stop the program
