import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import os
import random

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
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
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


# Flashcard Generation
def generate_flashcards(text_chunks, api_key):
    global flashcards
    # For simplicity, we'll generate basic flashcards from text chunks. This could be extended for more complex logic.
    sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    flashcards = []

    for chunk in text_chunks:
        question = f"What is discussed in this text: {chunk[:200]}?"
        answer = chunk[:300]  # Just for demonstration; could use a QA model to extract meaningful answers
        flashcards.append({"question": question, "answer": answer})

    # Rank flashcards by relevance (using simple sentence embeddings)
    questions = [fc["question"] for fc in flashcards]
    embeddings = sentence_bert_model.encode(questions, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(embeddings, embeddings).mean(axis=1)
    ranked_flashcards = [flashcards[i] for i in scores.argsort(descending=True)]

    return ranked_flashcards[:10]  # Limit to top 10 flashcards


def main():
    global pdf_docs, flashcards
    st.header("NOTEZ: AI Chatbot - Ask Questions from PDFs")

    # Sidebar
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
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
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            flashcards = generate_flashcards(text_chunks, api_key)
            st.success("Flashcards generated!")

    if flashcards:  # Check if flashcards exist before displaying them
        st.subheader("Generated Flashcards")
        for i, flashcard in enumerate(flashcards):
            st.markdown(f"**Flashcard {i+1}:**")
            st.write(f"**Question**: {flashcard['question']}")
            st.write(f"**Answer**: {flashcard['answer']}")

        st.subheader("Quiz Time!")
        flashcard = random.choice(flashcards)
        question = flashcard["question"]
        answer = flashcard["answer"]

        user_answer = st.text_input(f"Question: {question}", key="quiz_answer")
        if user_answer:
            if user_answer.strip().lower() == answer.strip().lower():
                st.success("Correct!")
            else:
                st.error(f"Incorrect! The correct answer is: {answer}")

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
