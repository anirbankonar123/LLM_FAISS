from pydantic import BaseModel
from enum import auto
from fastapi_utils.enums import StrEnum

class ModelName(StrEnum):
    openAI = auto()
    openhermes_mistral_7B = auto()
    gemini_pro = auto()
    palm_api_text_bison = auto()

class Metadata(BaseModel):
    pageNo:str=""
    doc:str=""
class Output(BaseModel):
    response:list[str]=[]
    responseTime:str=""
    status:str="success"
    metadata_list:list[Metadata]=[]
    errorCode:str="0"
    errorMsg:str=""

class OutputIngest(BaseModel):
    status:str="ingestion success"
    errorCode:str="0"
    errorMsg:str=""

class OutputRAG(BaseModel):
    response:list[str]=[]
    status:str="success"
    errorMsg:str=""


