# 📄 Chat with PDFs

An intelligent AI-powered chatbot that lets you **chat with your PDF documents** using Groq + RAG (Retrieval-Augmented Generation).

![Demo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 🚀 Live Demo

[**Try the App Here**](https://knehvzou4qo96ekr3xfoth.streamlit.app/)

---

## ✨ Features

- Upload multiple PDF files
- Smart text extraction (handles normal & complex PDFs)
- Semantic search (understands meaning, not just keywords)
- Fast responses using Groq (Llama 3.1)
- Shows debug information for transparency
- Clean and easy-to-use interface

- ## 📸 Screenshot

![Chat with PDFs](<img width="950" height="416" alt="Screenshot 2026-06-30 233733" src="https://github.com/user-attachments/assets/49c753af-6532-46bc-8e3b-223429fee66e" />)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama-3.1-8B)
- **Embeddings**: sentence-transformers
- **Vector Store**: Chroma
- **PDF Processing**: PyMuPDF (fitz)

## How to Use

1. Go to the [Live App](https://knehvzou4qo96ekr3xfoth.streamlit.app/)
2. Upload your PDF files in the sidebar
3. Click **"Process PDFs"**
4. Start asking questions about the documents!

**Example Questions:**
- What is the name of the book?
- Summarize Chapter 3
- What is the ISBN number?
- Tell me about data visualization in this book

---

## Local Setup (For Developers)

```bash
git clone https://github.com/YOUR_USERNAME/chat-with-pdfs.git
cd chat-with-pdfs
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
streamlit run app.py
