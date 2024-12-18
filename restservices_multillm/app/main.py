import os
from fastapi import FastAPI, File, UploadFile
from typing import Union
import uvicorn
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from app import data_models
from utils import fileReader,validator, qa_util, search_util,ingest_util

app = FastAPI(description="Query App", version="0.1.0")

@app.get("/")
def read_root():
    return {"msg": "Hello World"}

model_kwargs = {"device":"cpu"}
model_name = "sentence-transformers/all-mpnet-base-v2"

embeddings = HuggingFaceEmbeddings(model_name = model_name, model_kwargs = model_kwargs)

#Initialize Vector DB with startup ingestion file
fileName = "IPCC_AR6_SYR_SPM.pdf"
if os.path.exists("faiss_index_pg"):
    knowledgeBase = FAISS.load_local("faiss_index_pg", embeddings)
    print("knowledge base loaded")
else:
    knowledgeBase = ingest_util.create_vectorDB(fileName,embeddings)
    knowledgeBase.save_local("faiss_index_pg")
    print("knowledge base created")


"""
This POST API accepts a Query and returns the response from a set of Private documents
The API takes optional parameters, modelName,temperature, top_p, max_tokens
"""
@app.post("/query",response_model=data_models.Output,summary="query a set of docs",description="This POST API accepts a query as a text \
The API takes optional parameters, modelName,temperature, top_p, max_tokens, repetition_penalty, top_k_RAG",response_description="The API returns the response of the Query, based on private documents \
", tags=["queryDocs"])
async def query_api(query:str, modelName: data_models.ModelName, temperature: Union[str, None] = "0.2",top_p: Union[str, None] = "0.4",max_tokens: Union[str, None] = "1024", repetition_penalty:Union[str,None]="1.0", top_k_RAG:Union[str,None]="1"):

    output = data_models.Output()
    output.status = "success"
    output.errorMsg = ""

    output = validator.validate(output,temperature,top_p,max_tokens,repetition_penalty,top_k_RAG)
    if (output.status=="failure"):
        return output
    query_response=""

    docs = search_util.search_vectordb(query,knowledgeBase,top_k_RAG)
    response_time=""

    metadata_list = []
    for i in range (len(docs)):
        Metadata = data_models.Metadata()
        Metadata.pageNo = docs[i].metadata['page']
        Metadata.doc = docs[i].metadata['doc']
        metadata_list.append(Metadata)

    try:
        query_response,response_time = qa_util.generate_answer(modelName,query,docs,temperature,top_p,max_tokens,repetition_penalty)

    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)


    output.response = query_response
    output.responseTime = response_time
    output.metadata_list = metadata_list

    return output
    
"""
This POST API accepts a User Story and returns the Test cases
The API takes optional parameters, modelName,temperature, top_p, max_tokens
"""
@app.post("/submitRequest/testcasegen",response_model=data_models.Output,summary="generate Test case from User story",description="This POST API accepts a User Story as text, and returns Test cases\
The API takes optional parameters, modelName,temperature, top_p, max_tokens, repetition_penalty, top_k_RAG",response_description="The API returns the test cases generated \
", tags=["submitTestCaseRequest"])
async def testcase_api(userstory:str, modelName: data_models.ModelName, temperature: Union[str, None] = "0.2",top_p: Union[str, None] = "0.4",max_tokens: Union[str, None] = "1024", repetition_penalty:Union[str,None]="1.0"):

    output = data_models.Output()
    output.status = "success"
    output.errorMsg = ""

    output = validator.validate(output,temperature,top_p,max_tokens,repetition_penalty,"1")
    if (output.status=="failure"):
        return output
    query_response=""

    response_time=""
    try:
        response_arr = []
        response_time_tot = 0.0
        reqmnt = userstory
        query_response, response_time = qa_util.generate_testcase(modelName, reqmnt, temperature, top_p, max_tokens,
                                                                  repetition_penalty)
        response_time_tot += float(response_time)
        response_arr.append(query_response)

    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)

    output.response = list(response_arr)
    output.responseTime = str(response_time)+" secs"

    return output

"""
This POST API accepts a csv File or Excel File with Requirements and writes the csv with generated test cases.
The API takes optional parameters, modelName,temperature, top_p, max_tokens
"""
@app.post("/submitBulkRequest/testcasegen",response_model=data_models.Output,summary="submit Test case request with a csv file",description="This POST API accepts a Excel File with Requirements \
The API takes optional parameters, modelName,temperature, top_p, max_tokens, top_k_RAG, num_scenarios", response_description="The API returns the status, with Folder Name, \
 ", tags=["submitBulkTestCaseRequest"])
async def submitBulkTestCaseRequest_api(modelName: data_models.ModelName, temperature: Union[str, None] = "0.2",top_p: Union[str, None] = "0.4",max_tokens: Union[str, None] = "1024", repetition_penalty:Union[str,None]="1.0", file: UploadFile = File(...)):


    output = data_models.Output()
    output.status = "success"
    output.errorMsg = ""

    print(file.content_type)

    if file.content_type not in ["text/csv","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        output.status = "failure"
        output.errorCode = "110"
        output.errorMsg = f"File type of {file.content_type} is not supported"
        return output

    output = validator.validate(output,temperature,top_p,max_tokens,repetition_penalty,"1")
    if (output.status=="failure"):
        return output

    file_name = file.filename.split(".")[0]+str(datetime.datetime.now())+".csv"
    filepath = os.path.join(".",file_name)
    print(filepath)

    try:
        #Convert uploaded file into pandas dataframe
        if (file.content_type=="text/csv"):
            df = fileReader.read_csv(file,filepath)
        else:
            contents = await file.read()
            file_arr = filepath.split(".")
            file_nm = file_arr[0]
            filepath = file_nm + ".csv"
            df = fileReader.read_excel(contents)
        # Write uploaded file into desired location
        print("No of records:" + str(len(df)))
        print(df.head())

    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to read uploaded File:"+str(error)

    try:
        response_arr=[]
        response_time_tot=0.0
        for i in range(len(df)):
            print("processing row no:"+str(i))
            reqmnt=str(df["Requirement"][i])
            query_response,response_time = qa_util.generate_testcase(modelName,reqmnt,temperature,top_p,max_tokens,repetition_penalty)
            response_time_tot+=float(response_time)
            response_arr.append(query_response)

        df["generated_testcase"]=response_arr
        df.to_csv(filepath,index=False)
        print("file written")
    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)

    output.response = list(response_arr)
    output.responseTime = str(response_time_tot)
    return output


"""
This POST API accepts a PDF file and returns status of ingestion
"""
@app.post("/ingest",response_model=data_models.OutputIngest,summary="ingest a PDF doc",description="This POST API accepts a PDF file\
", response_description="The API returns the status of ingestion", tags=["ingestdoc"])
async def ingest_api(file_data: UploadFile = File(...)):

    output = data_models.OutputIngest()
    output.status = "success"
    output.errorMsg = ""
    file_path="uploadDocs"
    filepath = os.path.join(file_path, file_data.filename)
    print(file_path)
    try:
        if not (os.path.exists(file_path)):
            os.mkdir(file_path)
        else:
            # Get the list of all files in the directory
            files = os.listdir(file_path)

            # Iterate over the list of files and delete each file
            for file in files:
                os.remove(os.path.join(file_path, file))
    except Exception as exc:
        print(
            f"Something went wrong in creating folder {file_path}. Error {exc}"
        )

    try:
        filepath = fileReader.read_doc(file_data, filepath)
        print(filepath)
        ingest_util.add_vectordb(filepath,knowledgeBase)
    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)


    return output

"""
This GET API returns the list of documents ingested
"""
@app.get("/listdocs",response_model=data_models.OutputRAG,summary="list the contents of RAG DB",description="This GET API lists the content of Vector DB", response_description="The API returns list of documents in VectorDB", tags=["listdocs"])
async def listdocs_api():

    output = data_models.OutputRAG()
    output.status = "success"
    output.errorMsg = ""

    try:
        listdocs = ingest_util.list_vectordb(knowledgeBase)
        print(listdocs)
        output.response = listdocs
    except Exception as error:
        output.status = "failure"
        output.errorCode = "100"
        output.errorMsg = "Failed to generate results:"+str(error)
    return output

if __name__ == "__main__":
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False, root_path="/")
