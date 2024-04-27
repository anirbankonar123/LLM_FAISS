FROM python:3.10
#FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt-get update
RUN apt-get install -y openssl
RUN pip3 install --upgrade --no-cache-dir pip

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
#RUN CUDACXX=/usr/local/cuda-11.8/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all-major" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

ENV OPENAI_API_KEY=<your openai key>

COPY ./app /code/app
COPY ./utils /code/utils
COPY ./*.json /code/
COPY ./openhermes-2.5-mistral-7b.Q4_0.gguf /code/
COPY ./IPCC_AR6_SYR_SPM.pdf /code/


EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

