import streamlit as st
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from io import BytesIO
from groq import Groq
import tempfile

# Try importing OCR libraries (optional)
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Chat with PDFs", page_icon="📄")

# Initialize Groq client
# Load API key from Streamlit secrets or .env
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    load_dotenv()
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize session state
if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("📄 Chat with your PDFs")

import fitz  # PyMuPDF - add this import at the top with other imports

def extract_text_from_pdf(pdf_file):
    """Improved extraction with detailed debugging"""
    try:
        pdf_bytes = pdf_file.getvalue()
        filename = pdf_file.name
        
        st.info(f"Processing: {filename}")
        
        # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        total_pages = len(doc)
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_text = page.get_text("text")
            if page_text and page_text.strip():
                text += f"Page {page_num+1}:\n{page_text}\n\n"
        
        doc.close()
        
        extracted_len = len(text.strip())
        
        with st.expander(f"📊 Extraction Debug - {filename}", expanded=True):
            st.write(f"**Total Pages:** {total_pages}")
            st.write(f"**Characters Extracted:** {extracted_len:,}")
            
            if extracted_len > 100:
                st.success("✅ Good extraction")
                preview = text[:800] + "..." if len(text) > 800 else text
                st.text_area("Preview of extracted text", preview, height=300)
            else:
                st.error("❌ Very little or no text extracted!")
                st.write("This PDF might be scanned or image-based.")
        
        if extracted_len > 200:
            return text.strip(), total_pages
        else:
            return None, total_pages
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0

def chunk_text(text, chunk_size=3000):
    """Split text into smaller chunks for API processing"""
    if not text:
        return []
    
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1
        if current_size + word_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_relevant_context(question, full_text, max_chunks=5):
    """Get most relevant chunks for the question"""
    chunks = chunk_text(full_text)
    
    if not chunks:
        return ""
    
    if len(chunks) <= max_chunks:
        return " ".join(chunks)
    
    # Simple keyword matching
    question_words = set(question.lower().split())
    chunk_scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        # Count how many question words appear in this chunk
        score = sum(1 for word in question_words if word in chunk_lower)
        # Bonus for longer chunks (more content)
        score += len(chunk) / 1000
        chunk_scores.append((score, i, chunk))
    
    # Sort by score and get top chunks
    chunk_scores.sort(reverse=True)
    top_chunks = [chunk for _, _, chunk in chunk_scores[:max_chunks]]
    
    return " ".join(top_chunks)

def get_all_document_text():
    """Combine text from all processed PDFs"""
    all_text = ""
    for filename, info in st.session_state.pdf_texts.items():
        if info.get('text'):
            all_text += f"\n=== Document: {filename} ===\n"
            all_text += info['text']
            all_text += "\n"
    return all_text

def answer_question(question, context):
    """Get answer from Groq LLM with debug info"""
    if not client:
        return "⚠️ Groq client not configured. Please check your API key."
    
    # DEBUG: Show context info
    with st.expander("🔍 Debug: Document Context"):
        st.write(f"📊 Total context length: {len(context):,} characters")
        st.write(f"📄 Context preview: {context[:500]}...")
        
        if len(context) < 100:
            st.error("⚠️ Context is too short! The PDF may not have been read correctly.")
        else:
            st.success(f"✅ Context loaded ({len(context):,} characters)")
    
    if not context or len(context.strip()) < 100:
        return "⚠️ No document content found. Please ensure your PDF has extractable text."
    
    try:
        # Get only relevant chunks
        relevant_context = get_relevant_context(question, context)
        
        # Show what we're sending to the LLM
        with st.expander("🔍 Debug: Sending to LLM"):
            st.write(f"📊 Relevant context length: {len(relevant_context):,} characters")
            st.write(f"📄 Preview: {relevant_context[:500]}...")
        
        if not relevant_context or len(relevant_context.strip()) < 100:
            return "⚠️ No relevant content found for your question. Try asking something more specific."
        
        # Ensure context doesn't exceed token limit
        if len(relevant_context) > 8000:
            relevant_context = relevant_context[:8000] + "... (truncated)"
        
        # Prepare prompt
        prompt = f"""You are a document assistant. Answer questions based ONLY on the provided context.

CONTEXT (from documents):
{relevant_context}

QUESTION: {question}

INSTRUCTIONS:
1. ONLY use information from the context above
2. If the answer isn't in the context, say "I cannot find that information in the documents"
3. Be specific and cite page numbers if mentioned
4. Keep answers concise and relevant

ANSWER:"""
        
        # Call Groq API
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful document assistant. Only answer using the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    
    if GROQ_API_KEY:
        st.success("✅ Groq API Key: Connected")
        st.info(f"🤖 Model: llama-3.1-8b-instant")
    else:
        st.error("❌ Groq API Key: Missing")
    
    # OCR Status
    if OCR_AVAILABLE:
        st.info("📷 OCR: Available (for scanned PDFs)")
    else:
        st.warning("⚠️ OCR: Not installed (scanned PDFs won't work)")
    
    uploaded_files = st.file_uploader(
        "📤 Upload your PDFs",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📎 {len(uploaded_files)} file(s) uploaded")
        
        if st.button("🔄 Process PDFs", type="primary"):
            with st.spinner("Processing PDFs..."):
                processed_count = 0
                for file in uploaded_files:
                    if file.name not in st.session_state.pdf_texts:
                        text, num_pages = extract_text_from_pdf(file)
                        
                        if text and len(text.strip()) > 50:
                            chunks = chunk_text(text)
                            st.session_state.pdf_texts[file.name] = {
                                "text": text,
                                "pages": num_pages,
                                "size": len(text),
                                "chunks": chunks,
                                "num_chunks": len(chunks)
                            }
                            if file.name not in st.session_state.processed_files:
                                st.session_state.processed_files.append(file.name)
                            processed_count += 1
                            st.success(f"✅ {file.name} ({num_pages} pages, {len(text):,} chars)")
                        else:
                            st.error(f"❌ {file.name}: No text extracted")
                
                if processed_count > 0:
                    st.success(f"✅ Successfully processed {processed_count} file(s)")
                else:
                    st.error("❌ No text could be extracted from any PDF")
        
        if st.session_state.processed_files:
            st.subheader("📚 Processed Documents")
            for filename in st.session_state.processed_files:
                info = st.session_state.pdf_texts[filename]
                with st.expander(f"📄 {filename}"):
                    st.write(f"📃 Pages: {info['pages']}")
                    st.write(f"📏 Characters: {info['size']:,}")
                    st.write(f"📦 Chunks: {info['num_chunks']}")
                    
                    # Show preview
                    preview = info['text'][:500] + "..." if len(info['text']) > 500 else info['text']
                    st.text_area("📖 Preview", preview, height=150, disabled=True)
                    
                    # Full text button
                    if st.button(f"🔍 Show full text for {filename}", key=f"full_{filename}"):
                        st.text_area("Full extracted text", info['text'], height=400, disabled=True)
    
    if st.button("🗑️ Clear All Documents"):
        st.session_state.pdf_texts = {}
        st.session_state.processed_files = []
        st.session_state.messages = []
        st.rerun()

# Main chat area
st.header("💬 Chat with your Documents")

if not st.session_state.processed_files:
    st.info("👈 Upload and process PDFs from the sidebar to start chatting!")
else:
    st.success(f"✅ {len(st.session_state.processed_files)} document(s) loaded. Ask questions below!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your PDFs..."):
    if not st.session_state.processed_files:
        st.warning("⚠️ Please upload and process PDFs first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                context = get_all_document_text()
                
                if len(context) < 100:
                    response = "⚠️ No document content found. Please ensure your PDF has extractable text."
                else:
                    response = answer_question(prompt, context)
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})