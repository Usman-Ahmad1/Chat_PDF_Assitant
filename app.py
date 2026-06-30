import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from io import BytesIO
from groq import Groq
import fitz  # PyMuPDF

# New imports for semantic search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try importing OCR libraries (optional)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ====================== API KEY & CLIENT SETUP ======================
load_dotenv()

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize Groq: {str(e)}")
        client = None
else:
    client = None
    st.warning("⚠️ GROQ_API_KEY not found!")

# ====================== Page Config ======================
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")

# Initialize embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()

# Session State
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("📄 Chat with your PDFs")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_bytes = pdf_file.getvalue()
        filename = pdf_file.name
        
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page_text = doc[page_num].get_text("text")
            if page_text.strip():
                text += f"Page {page_num+1}:\n{page_text}\n\n"
        
        doc.close()
        
        extracted_len = len(text.strip())
        
        with st.expander(f"📊 Debug: {filename}", expanded=False):
            st.write(f"**Pages:** {total_pages} | **Characters:** {extracted_len:,}")
            if extracted_len > 100:
                st.success("✅ Good extraction")
            else:
                st.error("❌ Poor extraction")
        
        return text.strip() if extracted_len > 200 else None, total_pages
        
    except Exception as e:
        st.error(f"Error reading {pdf_file.name}: {str(e)}")
        return None, 0

def create_vectorstore():
    """Create vector database from all processed documents"""
    all_chunks = []
    for filename, info in st.session_state.pdf_texts.items():
        if info.get('text'):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_text(info['text'])
            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": filename})
    
    if all_chunks:
        texts = [c["text"] for c in all_chunks]
        metadatas = [{"source": c["source"]} for c in all_chunks]
        
        st.session_state.vectorstore = Chroma.from_texts(
            texts=texts, 
            embedding=embeddings, 
            metadatas=metadatas
        )
        return True
    return False

def get_relevant_context(question, k=5):
    if st.session_state.vectorstore is None:
        return ""
    docs = st.session_state.vectorstore.similarity_search(question, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def answer_question(question, context):
    if not client:
        return "⚠️ Groq client not configured."
    
    if not context or len(context.strip()) < 100:
        return "⚠️ No relevant content found."
    
    try:
        prompt = f"""Answer based ONLY on the context below.

CONTEXT:
{context}

QUESTION: {question}

Answer concisely and cite source if possible."""

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful document assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    if GROQ_API_KEY:
        st.success("✅ Groq Connected")
    
    uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files and st.button("🔄 Process PDFs", type="primary"):
        with st.spinner("Processing..."):
            processed_count = 0
            for file in uploaded_files:
                if file.name not in st.session_state.pdf_texts:
                    text, pages = extract_text_from_pdf(file)
                    if text:
                        st.session_state.pdf_texts[file.name] = {"text": text, "pages": pages}
                        st.session_state.processed_files.append(file.name)
                        processed_count += 1
            
            if processed_count > 0:
                with st.spinner("Building search index..."):
                    if create_vectorstore():
                        st.success(f"✅ {processed_count} document(s) processed successfully!")

    # Show processed files
    if st.session_state.processed_files:
        st.subheader("Processed Documents")
        for fn in st.session_state.processed_files:
            st.write(f"✅ {fn}")

    if st.button("🗑️ Clear All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Chat Interface
st.header("💬 Ask Questions")

if not st.session_state.processed_files:
    st.info("Upload and process PDFs from sidebar to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = get_relevant_context(prompt)
                response = answer_question(prompt, context)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})