
# %%v
import dotenv
import os
dotenv.load_dotenv()

#langchain.debug = True

from langchain.document_loaders import CSVLoader 

API_KEY=os.environ["GOOGLE_API_KEY"]
csvLoader = CSVLoader(file_path='goodlock-manual.csv', encoding='utf-8')
docs = csvLoader.load()

#print(API_KEY)

from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings

os.environ["GOOGLE_API_KEY"] = API_KEY

ko_embedding= HuggingFaceBgeEmbeddings(
  model_name='./embedding_model',
)

vectordb = Chroma(
  persist_directory='./chroma2.db',
  embedding_function=ko_embedding,
)

retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":10, "fetch_k":20})


import google.generativeai as genai
#from google.generativeai.types import GenerationConfig

system_prompt = """
당신은 메뉴얼을 요약하여 답변을 해주는 고객센터 직원 입니다.
질문에 답하기 위해 검색된 메뉴얼을 사용하세요
기존에 대화한 이력이 있다면 대화 이력을 메뉴얼 보다 우선하여 응답합니다.
메뉴얼에 없는 내용은 지어내지 말고 모른다고 답변해야 합니다.
사용자 질문중 추노마크는 네트워크 아이콘의 은어 입니다.
답변은 세 문장 이내로 간결하게 유지하고, 일관된 톤과 형식으로 답변합니다.
사용자 질문중 추노마크는 통신사(sk, lg, kt 등) 아이콘의 은어 입니다. 네트워크 아이콘이라고 합니다.
당신은 따라서 답변할 때 추노마크 라는 말을 사용하면 안됩니다.
답변할 때에는 마지막에 기능을 사용할 때 필요한 앱의 package 를 적어주세요. 없으면 빈칸으로, 예를 들면

#답변 예시
* query : 통신사 아이콘을 지우고 싶어
* answer :
  QuckStar(퀵스타)를 통해 상단바에 보이는 아이콘을 숨길 수 있어요.
  QuickStar > 아이콘 표시를 선택하고 목록에서 원하는 항목을 끄면 됩니다.
  package=com.samsung.android.qstuner

* query : 굿락이 뭐야
* answer :
  메뉴얼 요약
  package=
"""

model = genai.GenerativeModel(
  model_name='gemini-1.5-flash',
  safety_settings=None,
  system_instruction=system_prompt,
  generation_config=genai.GenerationConfig(
     temperature=0.2,
  )
)


#db = Chroma()
# %%

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserQuery(BaseModel):
    content: str
    histories: list[dict]

class Result(BaseModel):
    answer: str


@app.get("/")
def main():
   return 'hello'

@app.post("/prompt", response_model=Result)
def requestPrompt(query: UserQuery):
  question = query.content
  history = query.histories
  print(f'hmhm prompt {question}')
  print(f'hmhm history {history}')

  docs = retriever.get_relevant_documents(question)
  docs= '\n'.join(map(lambda x: x.page_content, docs))

  messages:list[dict] = []
  messages.append({'role' : 'user',
                   'parts' : [f'#메뉴얼 {docs}']})
  messages.extend(history)
  messages.append({'role' : 'user',
                   'parts' : [f'query: {question}']})
  
  #print(f'hmhmhm message {messages}')
  try:
    response = model.generate_content(messages)
    return Result(answer=response.text)
  except Exception as e:
    print(f'{type(e).__name__}: {e}')
    return ''

# %%
# question = '추노마크가 뭐야'
# docs = retriever.get_relevant_documents(question)
# docs= '\n'.join(map(lambda x: x.page_content, docs))

# messages:list[dict] = []
# messages.append({'role' : 'user',
#                   'parts' : [docs]})
# messages.extend([])
# messages.append({'role' : 'user',
#                   'parts' : [f'query: {question}']})

# response = model.generate_content(messages)
# print(response.text)

# %%
