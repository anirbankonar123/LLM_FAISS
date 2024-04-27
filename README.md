# LLM Comparison API 
This repo provides a REST API system to do querying on private documents (/query endpoint) ingested using the /ingest endpoint into a local FAISS index <br> 
The query endpoint providse the facility to generate results using one of the 4 LLMs:<br>

openai (gpt-3.5-turbo)<br>
openhermes-mistral-7B (4-bit quantized)<br>
Google gemini-pro<br>
Google text-bison<br>

# Getting started

```
git clone git@github.com:anirbankonar123/LLM_FAISS.git

```
## Pre-requisites

python 3.10<br>
openai==0.28.1<br>
langchain==0.0.316<br>
langchain_community<br>
fastapi<br>
pyPDF<br>
python-multipart<br>
uvicorn<br>
fastapi_utils<br>
transformers['torch']<br>
faiss-gpu<br>
sentence-transformers<br>
sentencepiece<br>
llama_cpp<br>
llama-cpp-python (#CUDACXX= CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade)<br>
google-generativeai<br>
google-cloud-aiplatform<br>

Steps to install cuda: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network <br>
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions<br>

FAISS Ref: https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html<br>
https://python.langchain.com/docs/integrations/vectorstores/faiss/ <br>

VectorDB Ref: https://thedataquarry.com/posts/vector-db-1/#location-of-headquarters-and-funding<br>

1. To deploy REST Endpoint locally 
```
export OPENAI_API_KEY=
cd restservices_multillm<br>
uvicorn app.main:app --reload
```
Open the app/main.py in any editor and supply a intial PDF file you want to ingest in fileName=..<br>
Open the utils/qa_util.py in any editor and supply the Google Vertex AI Service Account json file path in key_path=...<br>
Provide the palm api key in palm.configure(api_key=...<br>
Provide the Vertex AI project id in PROJECT_ID=<br>

Go to localhost:8000/docs to test REST Endpoints using Swagger interface

2. To build docker image and deploy REST Endpoint:
```
docker build . -t rag-compare -f deploy-container.dockerfile
docker run --name ragcompare -d -p 8000:8000 rag-compare
docker ps
```
Go to 0.0.0.0:8000/docs to test REST Endpoints using Swagger interface