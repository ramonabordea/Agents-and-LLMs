import streamlit as st
import os
import torch 
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

def main():
    st.title("PDF Q&A Assistant")
    
    # PDF Upload
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
    
    # Question Input
    user_question = st.text_input("Ask a question about your PDFs")
    
    if st.button("Get Answer"):
        if uploaded_files and user_question:
            # Process PDFs (similar to your existing code)
            pdf_texts = []
            pdf_filenames = []
            
            for uploaded_file in uploaded_files:
                pdf_text = extract_text_from_pdf(uploaded_file)
                pdf_texts.append(pdf_text)
                pdf_filenames.append(uploaded_file.name)
            
            # Initialize embedding and vectorstore
            embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vectorstore = Chroma(
                embedding_function=embedding_model,
                collection_name="pdf_documents",
                persist_directory="./chromadb"
            )
            
            # Add documents to vectorstore
            for i, text in enumerate(pdf_texts):
                vectorstore.add_texts(texts=[text], ids=[pdf_filenames[i]])
            
            # Query and get response
            results = vectorstore.similarity_search(query=user_question, k=3)
            
            if results:
                # Construct prompt
                prompt = "You are an assistant. Use the following documents to answer the question:\n"
                for result in results:
                    prompt += f"{result.page_content}\n"
                prompt += f"\nAnswer the following question: {user_question}"
                
                # Get OpenAI response
                openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Display results
                st.subheader("Answer:")
                st.write(response.choices[0].message.content)
                
                st.subheader("Retrieved Documents:")
                for i, result in enumerate(results, 1):
                    st.text(f"Document {i}: {result.page_content[:200]}...")
            else:
                st.warning("No relevant information found.")

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

if __name__ == "__main__":
    main()
