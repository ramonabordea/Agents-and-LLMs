import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import openai
import streamlit as st
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Set the environment variable to allow duplicate OpenMP libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Debugging: Print the API key (ensure this is safe to do)
#t.write("OpenAI API Key:", os.getenv('OPENAI_API_KEY'))  # This should not print the actual key in production

# Streamlit app
st.title("Let's have fun with LLM")

# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB client and collection using LangChain's Chroma
vectorstore = Chroma(
    embedding_function=embedding_model,
    collection_name="pdf_documents",
    persist_directory="./chromadb"
)

# Initialize OpenAI client

# Example user question
user_question = st.text_input("Enter your question:")

if user_question:
    # Query ChromaDB using the proper LangChain API
    results = vectorstore.similarity_search(query=user_question, k=3)  # k=3 means top 3 results
    st.write(f"**User Question:** {user_question}")  # This line prints the user question
    # Process and generate a response using OpenAI
    if results:
        prompt = "You are an assistant. Use the following documents to answer the question:\n"
        for result in results:
            prompt += f"{result.page_content}\n"
        prompt += f"\nAnswer the following question: {user_question}"

        # Get the response from the OpenAI model
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract and display the response
        if response.choices:
            st.write(f"**Answer:** {response.choices[0].message.content}")
        else:
            st.write("No response generated.")
    else:
        st.write("No relevant information found.")
