import streamlit as st
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import logging

logging.basicConfig(level=logging.INFO)

def load_llm(model_name):
    llm = CTransformers(
        model=model_name,
        model_type="mistral",
        max_new_tokens=1048,
        temperature=0
    )
    return llm

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    content = ''
    for page in data:
        content += page.page_content
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30
    )

    chunks = splitter.split_text(content)
    documents = [Document(page_content=t) for t in chunks]
    return documents


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def llm_pipeline(file_path, model_name):
    documents = file_processing(file_path)
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_documents(documents, embeddings)
    llm_answer_gen = load_llm(model_name)
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                          chain_type="stuff", 
                                                          retriever=vector_store.as_retriever())
    return answer_generation_chain

def run_app():
    st.title("Question over PDF using HF")

    model_selection = st.selectbox(
        'Select a Model:',
        ('TheBloke/Mistral-7B-Instruct-v0.1-GGUF', 'TheBloke/zephyr-7B-alpha-GGUF')
    )

    uploaded_file = st.file_uploader("Upload your PDF file here", type=['pdf'])

    if uploaded_file:
        with st.spinner("Analyzing..."):
            with open("temp_pdf.pdf", "wb") as f:
                f.write(uploaded_file.getvalue())
            answer_generation_chain = llm_pipeline("temp_pdf.pdf", model_selection)

        st.success("PDF Analyzed! You can now ask questions.")

        question = st.text_input("Posez votre question ici")

        if st.button("Ask"):
            with st.spinner("Fetching answer..."):
                response = answer_generation_chain.run(question)
                st.write(response)

run_app()
