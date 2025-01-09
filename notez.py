import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import os
import torch
import nltk
from concurrent.futures import ThreadPoolExecutor

# Download NLTK tokenizer
nltk.download('punkt')

# Streamlit Page Configuration
st.set_page_config(page_title="NOTEZ: A Book GPT with Flashcards", layout="wide")

# Sidebar: Title and Explanation
with st.sidebar:
    st.title("NOTEZ: A Book GPT")
    st.markdown(
        """
        ## How It Works

        - *Enter Your API Key*: Obtain a Google API key at [Google Makersuite](https://makersuite.google.com/app/apikey).
        - *Upload Your PDFs*: Upload multiple documents for analysis.
        - *Process Files*: Extract and preprocess the content from your PDFs.
        - *Ask Questions*: Query your documents for detailed insights.
        - *Generate Flashcards*: Extract text and create flashcards based on key points.

        This project uses advanced language models for text processing.
        """
    )

# Sidebar: API Key Input and Process Button
with st.sidebar:
    st.header("API & Process")
    api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

    pdf_docs = st.file_uploader("Upload your PDF File:", accept_multiple_files=True, key="pdf_uploader")

    if st.button("Submit & Process", key="process_button") and api_key:
        with st.spinner("Processing..."):
            try:
                raw_text = "".join([
                    PdfReader(pdf).pages[i].extract_text()
                    for pdf in pdf_docs
                    for i in range(len(PdfReader(pdf).pages))
                ])
                text_chunks = RecursiveCharacterTextSplitter(
                    chunk_size=10000, chunk_overlap=1000
                ).split_text(raw_text)
                st.success("Processing Complete! Your document is ready.")
            except Exception as e:
                st.error(f"An error occurred while processing: {str(e)}")

# Tabs: Organize functionality into separate tabs
st.header("NOTEZ Functionalities")
tabs = st.tabs(["PDF Viewer", "Chatbot", "Flashcards"])

# PDF Viewer Tab
with tabs[0]:
    st.subheader("Uploaded PDF Viewer")
    if pdf_docs:
        for pdf in pdf_docs:
            with st.expander(f"{pdf.name}"):
                pdf_reader = PdfReader(pdf)
                for i, page in enumerate(pdf_reader.pages):
                    st.text(f"Page {i + 1}")
                    st.write(page.extract_text())

# Chatbot Tab
with tabs[1]:
    st.subheader("Chatbot")
    user_question = st.text_input("Ask a Question from the PDFs", key="user_question")
    if user_question and api_key:
        try:
            st.write("Reply: Feature Coming Soon")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Flashcards Tab
with tabs[2]:
    st.subheader("Flashcards")
    if pdf_docs:
        deck_name = st.text_input("Enter the name of the flashcard deck:", key="deck_name")
        flashcard_limit = st.number_input(
            "Set the flashcard limit (default is 10):", min_value=1, max_value=50, value=10
        )

        if st.button("Generate Flashcards", key="generate_flashcards_button"):
            with st.spinner("Generating flashcards..."):
                try:
                    os.makedirs("decks", exist_ok=True)

                    # Load models
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    t5_model = T5ForConditionalGeneration.from_pretrained("lmqg/t5-large-squad-qg").to(device)
                    t5_tokenizer = T5Tokenizer.from_pretrained("lmqg/t5-large-squad-qg")
                    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)
                    sentence_bert_model = SentenceTransformer('all-MiniLM-L6-v2')

                    def generate_questions_and_answers(chunk):
                        inputs = t5_tokenizer.encode(
                            "generate question: " + chunk, return_tensors="pt", truncation=True, max_length=512
                        ).to(device)
                        outputs = t5_model.generate(
                            inputs, max_length=64, num_beams=4, early_stopping=True
                        )
                        question = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

                        answer = qa_pipeline(question=question, context=chunk).get("answer", "No answer found.")

                        # Validate meaningful questions and answers
                        if question.strip() and answer.strip() and question.lower() != answer.lower():
                            return {"question": question, "answer": answer}
                        return None

                    flashcards = []
                    for pdf in pdf_docs:
                        pdf_reader = PdfReader(pdf)
                        pdf_text = "".join([page.extract_text() for page in pdf_reader.pages])
                        text_chunks = RecursiveCharacterTextSplitter(
                            chunk_size=500, chunk_overlap=100
                        ).split_text(pdf_text)

                        with ThreadPoolExecutor() as executor:
                            results = executor.map(generate_questions_and_answers, text_chunks)

                        for result in results:
                            if result:
                                flashcards.append(result)

                    # Rank flashcards by relevance
                    questions = [fc["question"] for fc in flashcards]
                    embeddings = sentence_bert_model.encode(questions, convert_to_tensor=True)
                    scores = util.pytorch_cos_sim(embeddings, embeddings).mean(axis=1)
                    ranked_flashcards = [flashcards[i] for i in scores.argsort(descending=True)[:flashcard_limit]]

                    output_path = f"decks/{deck_name}_flashcards.md"
                    with open(output_path, "w") as f:
                        for card in ranked_flashcards:
                            f.write(f"### Flashcard\n*Question: {card['question']}\nAnswer*: {card['answer']}\n\n")

                    st.success(f"Flashcards saved to {output_path}")

                    # Provide download link for the markdown file
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Flashcards Markdown",
                            data=f,
                            file_name=f"{deck_name}_flashcards.md",
                            mime="text/markdown"
                        )

                except Exception as e:
                    st.error(f"An error occurred during flashcard generation: {str(e)}")
