from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
from langchain.prompts import PromptTemplate


# Prompt
# prompt = PromptTemplate.from_template(
#     """
#     ### System:\n
# You are a helpful code assistant. Your task is to provide a test scenario for the below feature & requirement mentioned
# Provide the Testing scenarios in bulleted numbers serially as shown in the example \n
#
# ### User:\n
# Feature: {feature_text_oneshot}
# Scenario: {reqmnt_text_oneshot}
# Test scenarios: {LLM_Res}
#
#     ### Assistant\n
#     Feature: {feature_text}
#     Scenario:{reqmnt_text}
#     Test scenarios :
#
#     """
# )

prompt = PromptTemplate.from_template(
    """[INST] \
    <<SYS>> \
    You are a helpful assistant to a Sofware Quality Assurance Engineer. \
    As part of the job, I read through the feature descriptions and a \
    requirement then generates test cases .\

    The feature description is: \
    {feature_text_oneshot} \
    The requirement is:\
    {reqmnt_text_oneshot}\
    Test scenarios are:\
    {LLM_Res}\ 
    [/INST]\
    The feature description is: \
    {feature_text} \
    The requirement is:\
    {reqmnt_text}\
    Test scenarios are:\

     <</SYS>>"""
)

model_id = "teknium/OpenHermes-2.5-Mistral-7B" #"Intel/neural-chat-7b-v3-1"
#

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(device)

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

time_1 = time()
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='cuda:0',
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
time_2 = time()
print(f"Prepare model, tokenizer: {round(time_2-time_1, 3)} sec.")

time_1 = time()
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
time_2 = time()
print(f"Prepare pipeline: {round(time_2-time_1, 3)} sec.")

def test_model(tokenizer, pipeline, prompt_to_test):
    """
    Perform a query
    print the result
    Args:
        tokenizer: the tokenizer
        pipeline: the pipeline
        prompt_to_test: the prompt
    Returns
        None
    """
    # adapted from https://huggingface.co/blog/llama2#using-transformers
    time_1 = time()
    sequences = pipeline(
        prompt_to_test,
        temperature=1.5,
        repetition_penalty=.5,
        do_sample=True,
        top_p=0.5,
#        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1024,)
    time_2 = time()
    print(f"Test inference: {round(time_2-time_1, 3)} sec.")
    result=""
    for seq in sequences:
        result+=seq['generated_text']
        #print(f"Result: {seq['generated_text']}")
    return result

feature_text_oneshot = "The Analytics functionality will provide a more precise, extended and relevant insights based on the information products (documents, articles...etc) reachable by the IKM Tools. It is targeted to the IKM arena and does not foresee the analysis of other data like FASes internal data types. This functionality will provide analytical services to support the decision-making needs of the enterprise, using information produced by gathering, consolidating, crossreferencing and enhancing information from various sources. Different types of analytics can be applied: Descriptive analytics looks at past performance and understands that performance by mining historical data to look for the reasons behind past success or failure. Predictive analytics is an area of data mining that deals with extracting information from data and using it to predict trends and behaviour patterns. It is trying to answer the question what will happen. The Analytics Services enable the development, management, generation and dissemination of reports from identified information sources in a format most readily understood by the target reader and possibly based on specified templates. The IKM Tools will support processing of structured and unstructured information, as streams or stored in persistent stores like databases. There is a dedicated data scientist team that produces the analytic reports based on the Microsoft Power BI and KNIME tools. The Contractor will make available their output into the IKM Tools. The IKM Tools will allow to publish analysed results in a variety of forms, as web content, MS Office documents and database stores"
reqmnt_text_oneshot = "The IKM Tools shall leverage the current Microsoft Power BI and KNIME technologies to provide the analytic services"
LLM_Res = """Based on the feature description and requirement provided, here are some test cases in Gherkins format for the Analytics functionality:
    
1. As an IKM Engineer, I want to be able to access the Analytics functionality within the IKM Tools so that I can utilize it to make more precise decisions about our enterprise's information products.

Feature: Access Analytics Functionality
  Scenario: Success Path
    Given the IKM Engineer is logged in to the IKM Tools
    When the Analytics functionality is accessed
    Then the Analytics dashboard should be displayed with relevant insights and analytics options
    
2. As an IKM Engineer, I want to be able to view descriptive analytics reports that analyze past performance of information products so that I can understand why they were successful or not.

Feature: Descriptive Analytics Reports
  Scenario: Success Path
    Given the IKM Engineer is logged in and accesses the Analytics functionality
    When a descriptive analytics report is requested for a specific information product
    Then the report should be generated and displayed with relevant insights and recommendations
    
3. As an IKM Engineer, I want to be able to view predictive analytics reports that forecast trends and behavior patterns of information products so that I can make informed decisions about their future success.

Feature: Predictive Analytics Reports
  Scenario: Success Path
    Given the IKM Engineer is logged in and accesses the Analytics functionality
    When a predictive analytics report is requested for a specific information product
    Then the report should be generated and displayed with relevant insights and recommendations
    
4. As an IKM Engineer, I want to be able to customize the templates used for generating analytic reports so that they are tailored to my specific needs and preferences.

Feature: Customizable Templates
  Scenario: Success Path
    Given the IKM Engineer is logged in and accesses the Analytics functionality
    When a template is selected or created for an analytic report
    Then the template should be applied to the report and displayed with relevant insights and recommendations
    
5. As an IKM Engineer, I want to be able to publish analyzed results in a variety of forms, such as web content, MS Office documents, and database stores so that they can be easily accessed by different users and systems.

Feature: Variety of Output Formats
  Scenario: Success Path
    Given the IKM Engineer is logged in and has accessed an analytic report
    When a request to publish the results is made
    Then the results should be published in the desired format (web content, MS Office documents, or database stores) and easily accessible by other users and systems."""
feature_text = "search"
reqmnt_text = "EDMS shall allow the user to get search results categorized by at least these domains: Enterprise Domain Command Domain Office Document Type and TT+ domain (i.e.: via filters refiners or scope searches)"
prompt_text = prompt.format(feature_text_oneshot=feature_text_oneshot, reqmnt_text_oneshot=reqmnt_text_oneshot, LLM_Res=LLM_Res, feature_text=feature_text, reqmnt_text=reqmnt_text)

print(test_model(tokenizer,
           query_pipeline,
           prompt_text))