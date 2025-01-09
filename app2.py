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
st.set_page_config(page_title="NOTEZ: Chatbot, Flashcards & Quiz", layout="wide")

st.markdown("""
## NOTEZ Chatbot, Flashcards & Quiz

NOTEZ combines multiple features to help you extract knowledge from your PDFs, interact with them through a chatbot, and generate flashcards for quiz preparation.

### Instructions

1. **Enter Your API Key**: You'll need a Google API key to access Google's Generative AI models.
2. **Upload PDF Files**: Upload multiple PDFs for processing.
3. **Ask Questions**: Query the PDFs for detailed answers via the chatbot.
4. **Generate Flashcards**: Generate flashcards for quiz preparation.
5. **Take Quiz**: Answer questions from the generated flashcards.
""")

# API key input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Global variables
pdf_docs = None
flashcards = []  # Initialize the flashcards variable

# PDF Text Extraction
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


def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


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

# Quiz Logic (from the provided code)
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
    global pdf_docs, flashcards
    st.header("NOTEZ: AI Chatbot - Ask Questions from PDFs")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    raw_text += extract_text_from_pdf(pdf)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Documents processed and vector store created.")

    # Chatbot Tab
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    # Flashcard & Quiz Tab
    st.subheader("Flashcards & Quiz")
    if st.button("Generate Flashcards"):
        with st.spinner("Generating flashcards..."):
            raw_text = ""
            for pdf in pdf_docs:
                raw_text += extract_text_from_pdf(pdf)

            questions = generate_questions(raw_text)
            refined_questions = filter_questions(questions)  # Refine the questions after generation

            # Get answers for these questions if API key is provided
            flashcard_answers = get_flashcard_answers(refined_questions, api_key)
            
            # Display Flashcards (questions and answers)
            st.write("Flashcards:")
            for idx, (question, answer) in enumerate(zip(refined_questions, flashcard_answers)):
                with st.expander(f"Flashcard {idx+1}"):
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {answer}")

    # PDF Viewer
    if pdf_docs:
        st.subheader("PDF Viewer")
        for pdf in pdf_docs:
            with st.expander(f"{pdf.name}"):
                pdf_reader = PdfReader(pdf)
                for i, page in enumerate(pdf_reader.pages):
                    st.text(f"Page {i + 1}")
                    st.write(page.extract_text())

if __name__ == "__main__":
    main()
