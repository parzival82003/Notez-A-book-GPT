import streamlit as st
import time
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from google.api_core.exceptions import ResourceExhausted
import pdfplumber
import os
import re

# Streamlit Page Configuration
st.set_page_config(page_title="Notez: AI Assistant", layout="wide")

st.markdown("""
## Notez

This tool allows you to upload PDF files, interact with them using AI, ask questions, or generate flashcards and quizzes from the text.
""")

# API key input for Chatbot
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# 1. **PDF Upload and Text Extraction**
uploaded_pdf = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

# 2. **File Size Check**
def is_file_size_within_limit(uploaded_file, size_limit_mb=200):
    file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
    return file_size <= size_limit_mb

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Chatbot Logic
def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Quiz Logic
def process_text_for_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Initialize the Hugging Face pipeline for Question Generation
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qa-qg-hl")

def generate_questions(text):
    chunks = process_text_for_chunks(text)
    questions = []
    for chunk in chunks:
        generated_questions = question_generator(chunk)
        questions.extend([q['generated_text'] for q in generated_questions])
    return questions

# Post-Processing: Filter out incomplete or vague questions
def filter_questions(questions):
    refined_questions = []
    for question in questions:
        if len(question) > 5 and not question.endswith(('will', 'is', 'it', 'are')):
            question = re.sub(r"(What is the name of the .+? that)", "What is the role of", question)
            question = re.sub(r"(What is the .+? process)", "What process ensures", question)
            question = question.replace("proprietary protocols and proprietary protocols", "What is the difference between proprietary protocols?")
            refined_questions.append(question.strip())
    return refined_questions

# Function to get answers for flashcard questions using the chatbot with retries
def get_flashcard_answers(questions, api_key, retries=3, delay=5):
    answers = []
    for question in questions:
        attempts = 0
        while attempts < retries:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(question)
                chain = get_conversational_chain(api_key)
                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                answers.append(response["output_text"])
                break
            except ResourceExhausted as e:
                attempts += 1
                st.warning(f"API quota exceeded. Retrying ({attempts}/{retries})...")
                time.sleep(delay)  # Wait before retrying
            except Exception as e:
                st.error(f"An error occurred: {e}")
                break
    return answers

# Frontend Display
def main():
    st.header("Notez: Flashcards & Quiz Mode")

    refined_questions = []  # Initialize variable to avoid UnboundLocalError

    raw_text = st.text_area("Paste your text here for question generation", height=200)

    if uploaded_pdf:
        # Ensure the uploaded PDFs are within size limits
        if all(is_file_size_within_limit(pdf) for pdf in uploaded_pdf):
            # Extract text from uploaded PDFs
            extracted_text = ""
            for pdf in uploaded_pdf:
                extracted_text += extract_text_from_pdf(pdf)

            if extracted_text:
                raw_text = extracted_text  # If PDFs are uploaded, use the extracted text
                st.text_area("Text extracted from PDFs", raw_text, height=200)

    # Check if there is text for processing
    if raw_text:
        # Generate questions from the provided text
        questions = generate_questions(raw_text)
        refined_questions = filter_questions(questions)  # Refine the questions after generation
        st.write("Generated Questions:")
        for idx, question in enumerate(refined_questions):
            st.write(f"{idx+1}. {question}")

        # Get answers for these questions if API key is provided
        if api_key:
            if st.button("Generate Flashcards"):
                with st.spinner("Generating flashcards..."):
                    flashcard_answers = get_flashcard_answers(refined_questions, api_key)

                    # Display Flashcards (questions and answers)
                    st.write("Flashcards:")
                    for idx, (question, answer) in enumerate(zip(refined_questions, flashcard_answers)):
                        with st.expander(f"Flashcard {idx+1}"):
                            st.write(f"**Q:** {question}")
                            st.write(f"**A:** {answer}")
    else:
        st.write("Please provide raw text for question generation.")

if __name__ == "__main__":
    main()
