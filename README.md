This repo provides a REST API system to do querying on private documents (/query endpoint) ingested using the /ingest endpoint REST API <br>
The endpoints provide the facility to generate results using one of the 4 LLMs:
openai (gpt-3.5-turbo)
openhermes-mistral-7B (4-bit quantized)
Google gemini-pro
Google text-bison

Sample csv is provided for ref.

Before running this, the following is needed:

Install the following dependencies:
openai==0.28.1
langchain==0.0.316
langchain_community
fastapi
pyPDF
python-multipart
uvicorn
fastapi_utils
transformers['torch']
faiss-gpu
sentence-transformers
sentencepiece
llama_cpp
llama-cpp-python (#CUDACXX= CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade)
google-generativeai
google-cloud-aiplatform

Steps to install cuda: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network <br>
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions<br>

FAISS Ref: langchain-ai/langchain#2699 https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html<br>

VectorDB Ref: https://thedataquarry.com/posts/vector-db-1/#location-of-headquarters-and-funding<br>

To run REST API: cd restservices
Open the app/main.py in any editor and supply a intial PDF file you want to ingest in fileName=..
Open the utils/qa_util.py in any editor and supply the Google Vertex AI Service Account json file path in key_path=...
Provide the palm api key in palm.configure(api_key=...
Provide the Vertex AI project id in PROJECT_ID=
export OPENAI_API_KEY=
Run the cmd:
uvicorn app.main:app --reload
Go to localhost:8000/docs to test REST Endpoints using Swagger interface
