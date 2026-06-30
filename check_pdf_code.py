import streamlit as st
from pypdf import PdfReader
import os

# Replace with your actual PDF filename
pdf_filename = "your_book.pdf"  # CHANGE THIS

try:
    reader = PdfReader(pdf_filename)
    print(f"📄 Total pages: {len(reader.pages)}")
    
    text = ""
    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text:
            text += page_text
            print(f"✅ Page {page_num}: {len(page_text)} characters")
        else:
            print(f"❌ Page {page_num}: NO TEXT EXTRACTED")
    
    print("\n" + "="*50)
    print(f"📊 Total characters extracted: {len(text)}")
    print("\n📖 First 500 characters:")
    print("-"*50)
    print(text[:500] if text else "⚠️ NO TEXT EXTRACTED!")
    
except FileNotFoundError:
    print(f"❌ File '{pdf_filename}' not found!")
    print("Please check the filename and try again.")
except Exception as e:
    print(f"❌ Error: {e}")