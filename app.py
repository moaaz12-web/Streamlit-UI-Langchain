import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import os
from dotenv import load_dotenv
import nltk
import streamlit as st
import pinecone
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

openai.api_key = os.getenv("OPENAI_API_KEY")

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

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(text)

        # Extract the store name from the PDF path
        store_name = os.path.splitext(os.path.basename(pdf_path))[0]
        st.write(f'{store_name}')

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                docsearch = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            pinecone.init(
            api_key=os.environ.get("PINECONE_API_KEY"),  # Replace with your Pinecone API key
            environment=os.environ.get("PINECONE_API_ENV")  # Replace with your Pinecone API environment
            )
            index_name = "moaaz"

            # Create the vectorstore using Pinecone
            docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)


            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(docsearch, f)

        # Accept user questions/query
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            qa = load_qa_chain(llm=OpenAI(model_name='text-davinci-002'))

            docs = docsearch.similarity_search(query, include_metadata=True)

            
            with get_openai_callback() as cb:
                response = qa.run( input_documents=docs,
                    question={
                              "query": f"Answer the following question in a humanly, friendly tone. Moreover, answer only in the language the question is asked in. Do not answer like a robot. Thoroughly understand the question and answer only in the specified language:\n{query}"
                    }
    )
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
