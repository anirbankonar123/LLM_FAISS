import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

def create_vectorDB(fileName,embeddings):
    loader = PyPDFLoader(fileName)

    pages = loader.load_and_split()

    print(len(pages))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    list_of_documents = []
    ctr=1
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            list_of_documents.append(Document(page_content=chunk, metadata=dict(page=ctr,doc=fileName)))
        ctr+=1

    knowledgeBase=FAISS.from_documents(list_of_documents,embeddings)

    return knowledgeBase

def update_doc(pages,fileName,knowledgeBase):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    list_of_documents = []
    ctr=1
    for page in pages:
        chunks = text_splitter.split_text(page.page_content)
        for chunk in chunks:
            list_of_documents.append(Document(page_content=chunk, metadata=dict(page=ctr,doc=fileName)))
        ctr+=1

    knowledgeBase.add_documents(list_of_documents)

    return knowledgeBase

def add_vectordb(fileName,knowledgeBase):
    loader = PyPDFLoader(fileName)

    pages = loader.load_and_split()

    print(len(pages))

    knowledgeBase = update_doc(pages, fileName, knowledgeBase)

    print("added new doc")

    knowledgeBase.save_local("faiss_index_pg")

def list_vectordb(knowledgeBase):
    docs=set()
    total_records = 0
    for key,document in knowledgeBase.docstore._dict.items():

        docs.add(document.metadata['doc'])
        total_records+=1
    print("Total documents in the collection: ", str(len(docs)))
    return list(docs)