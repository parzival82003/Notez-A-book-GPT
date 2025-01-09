import streamlit as st
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from google.api_core.exceptions import ResourceExhausted

# Streamlit Page Configuration
st.set_page_config(page_title="Flashcard Test", layout="wide")

# API key input for Chatbot
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

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

# Helper function to generate conversational chain
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

# Helper function for chunking text (for questions)
def process_text_for_chunks(text, chunk_size=1000):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# Frontend Display: Main logic for flashcards
def main():
    st.header("Test Flashcard Generation")

    # Input text for testing
    raw_text = st.text_area("Paste your text here for question generation", height=200)

    if raw_text:
        # Generate questions from the provided text
        question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qa-qg-hl")
        chunks = process_text_for_chunks(raw_text)
        questions = []
        for chunk in chunks:
            generated_questions = question_generator(chunk)
            questions.extend([q['generated_text'] for q in generated_questions])

        # Display generated questions
        st.write("Generated Questions:")
        for idx, question in enumerate(questions):
            st.write(f"{idx+1}. {question}")

        # Get answers for these questions if API key is provided
        if api_key:
            if st.button("Generate Flashcards"):
                with st.spinner("Generating flashcards..."):
                    flashcard_answers = get_flashcard_answers(questions, api_key)

                    # Display Flashcards (questions and answers)
                    st.write("Flashcards:")
                    for idx, (question, answer) in enumerate(zip(questions, flashcard_answers)):
                        with st.expander(f"Flashcard {idx+1}"):
                            st.write(f"**Q:** {question}")
                            st.write(f"**A:** {answer}")

if __name__ == "__main__":
    main()
