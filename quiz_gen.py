import streamlit as st
import PyPDF2
from transformers import pipeline
import re
import pdfplumber

# 1. **Streamlit Front-End Setup**
st.title("Quiz Generation from PDF")

# 2. **PDF Upload and Text Extraction**
uploaded_pdf = st.file_uploader("Upload your PDF", type="pdf")

# 3. **File Size Check**
def is_file_size_within_limit(uploaded_file, size_limit_mb=200):
    file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
    return file_size <= size_limit_mb

# 4. **Text Extraction using PyPDF2 and pdfplumber**
def extract_text_from_pdf(file):
    # Use pdfplumber for more reliable text extraction
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text

# 5. **Process Extracted Text for API Input**
def process_text_for_chunks(text, chunk_size=1000):
    chunks = []
    # Split text into chunks to avoid model input limit
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# 6. **Question Generation using Hugging Face API**
# Initialize the Hugging Face pipeline for Question Generation
question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qa-qg-hl")

def generate_questions(text):
    # Generate questions from text chunks
    chunks = process_text_for_chunks(text)
    questions = []
    for chunk in chunks:
        generated_questions = question_generator(chunk)
        questions.extend([q['generated_text'] for q in generated_questions])
    return questions

# 7. **Post-Processing: Filter out incomplete or vague questions**
def filter_questions(questions):
    refined_questions = []
    for question in questions:
        # Remove incomplete questions (those that end abruptly or contain irrelevant fragments)
        if len(question) > 5 and not question.endswith(('will', 'is', 'it', 'are')):
            # Remove redundant or vague phrases (adjust based on specific needs)
            question = re.sub(r"(What is the name of the .+? that)", "What is the role of", question)
            question = re.sub(r"(What is the .+? process)", "What process ensures", question)
            
            # Remove redundant phrasing or corrections
            question = question.replace("proprietary protocols and proprietary protocols", "What is the difference between proprietary protocols?")
            
            refined_questions.append(question.strip())
    return refined_questions

# 8. **Display Questions on Streamlit Front-End**
if uploaded_pdf:
    if is_file_size_within_limit(uploaded_pdf):
        # Extract text from uploaded PDF
        text = extract_text_from_pdf(uploaded_pdf)
        
        if text:
            # Generate questions from the extracted text
            with st.spinner('Generating questions, please wait...'):
                questions = generate_questions(text)
            
            # Post-process and filter out incomplete/vague questions
            refined_questions = filter_questions(questions)
            
            # Display the generated questions
            st.write("Generated Questions:")
            for idx, question in enumerate(refined_questions):
                st.write(f"{idx+1}. {question}")
        else:
            st.write("No text could be extracted from the PDF.")
    else:
        st.error("The uploaded PDF exceeds the 200MB size limit. Please upload a smaller file.")
else:
    st.write("Please upload a PDF to generate questions.")
