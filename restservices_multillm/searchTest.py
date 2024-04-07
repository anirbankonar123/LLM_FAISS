import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import langchain

print(langchain.__version__)

model_kwargs = {"device":"cuda"}
model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs)

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


def update_doc(pages,fileName):
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

fileName="/home/anish/Downloads/IPCC_AR6_SYR_SPM.pdf"
if os.path.exists("faiss_index_pg"):
    knowledgeBase = FAISS.load_local("faiss_index_pg", embeddings)
    print("knowledge base loaded")
else:
    knowledgeBase = create_vectorDB(fileName,embeddings)
    knowledgeBase.save_local("faiss_index_pg")
    print("knowledge base created")

def list_vectordb():
    docs=set()
    total_records = 0
    for key,document in knowledgeBase.docstore._dict.items():
        docs.add(document.metadata['doc'])
        total_records+=1
    print("Total documents in the collection: ", str(len(docs)))
    return docs

def add_vectordb(fileName):
    loader = PyPDFLoader(fileName)

    pages = loader.load_and_split()
    print(len(pages))
    knowledgeBase = update_doc(pages, fileName)
    print("added new doc")
    knowledgeBase.save_local("faiss_index_pg")


def search_vectordb(query):
    docs = knowledgeBase.similarity_search(query,1)
    print(docs[0])

    return docs


if __name__ == "__main__":
    search_vectordb("How much did the temperature change in 2011 compared to 1850 ?")
    #docs=list_vectordb()
    #print(docs)
    # add_vectordb("/home/anish/Downloads/Invoice-32646.pdf")
    # list_vectordb()
    #print(openai.__version__)
    #print(langchain.__version__)