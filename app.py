import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
import openai
import requests

load_dotenv()

from langchain.chains import VectorDBQA

def download_pdf():
    # Specify the URL of the PDF file to download
    pdf_url = "https://public.railinc.com/sites/default/files/documents/ShipmentConditions.pdf"

    # Send a GET request to download the PDF
    response = requests.get(pdf_url)

    # Save the PDF locally
    with open("downloaded_pdf.pdf", "wb") as f:
        f.write(response.content)

def main():
    st.header("Chat with PDF 💬")

    # Add a button to trigger the PDF update
    if st.button("Update PDF"):
        st.write("Downloading the PDF...")
        download_pdf()
        st.write("PDF has been downloaded.")
    # Load the downloaded PDF
    pdf_path = "downloaded_pdf.pdf"
    if not os.path.exists(pdf_path):
        st.write("PDF not found. Click the 'Update PDF' button to download.")
    else:
        pdf_reader = PdfReader(pdf_path)
       
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Extract the store name from the PDF path
        store_name = os.path.splitext(os.path.basename(pdf_path))[0]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
