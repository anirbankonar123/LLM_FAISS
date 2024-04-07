
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_cpp import Llama
from app import data_models
from time import time
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
import vertexai
import google.generativeai as palm
from google.api_core import client_options as client_options_lib
from vertexai.language_models import TextGenerationModel
from langchain.prompts import PromptTemplate
import google.generativeai as genai

key_path = <path to json key of Vertex AI project service account>'

credentials = Credentials.from_service_account_file(
    key_path,
    scopes=['https://www.googleapis.com/auth/cloud-platform'])

if credentials.expired:
    credentials.refresh(Request())

palm.configure(
    api_key="<api key of palm api>",
    transport="rest",
    client_options=client_options_lib.ClientOptions(
        api_endpoint="https://generativelanguage.googleapis.com/",
    )
)

models = [m for m in palm.list_models()
          if 'generateText'
          in m.supported_generation_methods]
model_bison = models[0]

PROJECT_ID = '<project id of vertex ai project>'
REGION = 'us-central1'
# initialize vertex
vertexai.init(project = PROJECT_ID, location = REGION, credentials = credentials)

# Prompt
prompt = PromptTemplate.from_template(
    """<|im_start|>system
    Answer the query with the context provided<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """
)

n_gpu_layers = -1
n_batch = 512
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = Llama(
    model_path="openhermes-2.5-mistral-7b.Q4_0.gguf",
    n_gpu_layers=n_gpu_layers,
    temperature=0.5,
    top_p=0.1,
    repetition_penalty=1.0,
    n_batch=n_batch,
    n_ctx=4096,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
)


def retrieve_answer_palm(prompt: str,temperature, top_p, max_tokens,repetition_penalty) -> str:
    text_response = []
    parameters = {
        "temperature": float(temperature),  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": int(max_tokens),  # Token limit determines the maximum amount of text output.
        "top_p": float(top_p),
        # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@002")
    response = model.predict(
        prompt,
        **parameters,
    )
    print(f"Response from Model: {response.text}")
    return "".join(response.text)

def make_rag_prompt(query, context):
  escaped = context.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative bot that answers questions using text from the context included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  QUESTION: '{query}'
  PASSAGE: '{context}'

  ANSWER:
  """).format(query=query,context=escaped)

  return prompt

def retrieve_answer_gemini(query: str,context:str,temperature, top_p, max_tokens,repetition_penalty) -> str:

    model = genai.GenerativeModel('gemini-pro')
    prompt = make_rag_prompt(query, context)
    response = model.generate_content(prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p))
    )

    return response.text


def retrieve_answer_openai(query,docs,temperature, top_p, max_tokens,repetition_penalty):
    model_kwargs = {"repetition_penalty": float(repetition_penalty)}
    response=""
    try:
        llm = OpenAI(model_name="gpt-3.5-turbo", temperature=float(temperature),top_p=float(top_p),max_tokens=int(max_tokens),presence_penalty=float(repetition_penalty)) #""gpt-3.5-turbo-16k-0613""
        chain = load_qa_chain(llm, chain_type='stuff')

        response = ""
        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)

    except Exception as err:
        print("exception in openai")
        print(str(err))

    return response


def retrieve_answer_mistral(input,temperature, top_p, max_tokens,repetition_penalty):

    result = llm(input,max_tokens=512) #llm_chain({'prompt': input})
    output = result["choices"][0]["text"]
    return output

def generate_answer(model_name,question,docs,temperature,top_p,max_tokens,repetition_penalty):
    time_1 = time()
    output_arr = []
    print(model_name)
    context_arr=[]
    context=""
    for doc in docs:
        context+="\n"+doc.page_content
    context_arr.append(context)

    if model_name == data_models.ModelName.openhermes_mistral_7B:
        print("generating response for model name mistral")
        for context in context_arr:
            input = f"""<|im_start|>system
            Answer the query with the context provided<|im_end|>
            <|im_start|>user
            question: {question} 
            context: {context}<|im_end|>
            <|im_start|>assistant
            """
            output = retrieve_answer_mistral(input,temperature, top_p, max_tokens,repetition_penalty)
            output_arr.append(output)

    elif model_name == data_models.ModelName.gemini_pro:
        print("generating response for model name gemini")
        for context in context_arr:

            output = retrieve_answer_gemini(question, context, temperature, top_p, max_tokens,repetition_penalty)
            output_arr.append(output)

    elif model_name == data_models.ModelName.openAI:
        print("generating response for model name openAI")
        output = retrieve_answer_openai(question, docs, temperature, top_p, max_tokens,repetition_penalty)
        output_arr.append(output)

    elif model_name == data_models.ModelName.palm_api_text_bison:
        print("generating response for model name palmAPI Text bison")
        for context in context_arr:
            input = f"question: {question} \n context: {context}"
            output = retrieve_answer_palm(input, temperature, top_p, max_tokens,repetition_penalty)
            output_arr.append(output)

    time_2 = time()
    print(f"Test inference: {round(time_2 - time_1, 3)} sec.")
    response_time = f"{round(time_2 - time_1, 3)} sec."
    print(output_arr)
    return list(output_arr),str(response_time)


