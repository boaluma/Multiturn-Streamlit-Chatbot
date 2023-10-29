import openai
# import fitz

import streamlit as st
from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import OpenAI 
# from langchain.chains.question_answering import load_qa_chain
# Set up your OpenAI API key

openai.api_key = os.environ["OPENAI_API_KEY"]
load_dotenv()

# app sidebars
with st.sidebar:
    st.title('PDF Chat App')
    st.markdown('''
    ## About
    A chat app that takes in PDF and answers user query based on the info in the PDF file uploded \n
    Made using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made with by Kunal Bitey')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    # doc = fitz.open(pdf_path)
    # text = ''
    # for page_num in range(doc.page_count):
    #     page = doc.load_page(page_num)
    #     text += page.get_text()
    # return text
    pdf_reader = PdfReader(pdf_path)
    text = ''
    for page in pdf_reader:
        text += page.extract_text()
    return text

# Initialize previous conversation
previous_conversation = []

def ask_question(question, context):
    global previous_conversation
    input_text = ''
    if previous_conversation:
        input_text = '\n'.join(f'Q: {q}\nA: {a}' for q, a in previous_conversation)
        input_text += f'\nQ: {question}\n'
    else:
        input_text = f'Q: {question}\n'

    input_text += context

    response = openai.Completion.create(
        engine="davinci", 
        prompt=input_text, 
        max_tokens=100
    )

    answer = response.choices[0].text.strip()
    previous_conversation.append((question, answer))
    return answer

def main():
    st.header("PDF Q&A app")
 
    # upload a PDF file
    pdf_path = st.file_uploader("Upload your PDF here", type='pdf')
    
    if pdf_path is not None:
        context = extract_text_from_pdf(pdf_path)
        
        question = st.text_input("Enter the question regarding your PDF :")
        
        if question:
            answer = ask_question(question, context)
            st.write(answer)
    
    # Example usage
    # question1 = "What is the capital of France?"
    # answer1 = ask_question(question1, context)
    # print(f"Answer: {answer1}")

    # question2 = "Who painted the Mona Lisa?"
    # answer2 = ask_question(question2, context)
    # print(f"Answer: {answer2}")
