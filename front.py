import streamlit as st
import requests
import re

st.title('GoodLock 챗봇')

if "messages" not in st.session_state:
  st.session_state.messages = []


for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["parts"][0], unsafe_allow_html=True)

def show_dlg():
  st.toast('준비 중입니다')

def extract_package(text):
  match = re.search(r'(.*)(package=\S*)', text)
  if match:
    result = text.split(r'package=')[0]
    extracted = match.group(2).split('=')[1]
  else:
    result = text
    extracted = None
  return result, extracted


# 프롬프트 비용이 너무 많이 소요되는 것을 방지하기 위해
MAX_MESSAGES_BEFORE_DELETION = 4
URL = 'http://localhost:8000/prompt'

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("질문 해주세요!"):

    # 만약 현재 저장된 대화 내용 기록이 4개보다 많으면 자르기
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        # Remove the first two messages
        del st.session_state.messages[0]
        del st.session_state.messages[0] 
    
    with st.chat_message("user"):
      st.markdown(prompt)
    
    answer = ''
    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      with st.spinner(text="답변 작성중입니다"):
        response = requests.post(
          URL,
          json={'content': prompt, 'histories': st.session_state.messages}
        )

        if (response.status_code == 200):
          answer = response.json()['answer']
          answer, package_name = extract_package(answer)
          answer.replace("\n", "<br>")
          message_placeholder.markdown(answer, unsafe_allow_html=True)
          print(package_name)

          if package_name != None:
            st.button(f'{package_name}을 설치할까요?', key=1, on_click=show_dlg)
            st.button(f'GTS preset1', key=2, on_click=show_dlg)
            st.button(f'GTS preset2', key=3, on_click=show_dlg)
        else:
          message_placeholder.markdown(f'응답 실패({response.status_code})')


    # with st.chat_message("assistant"):
    #   message_placeholder = st.empty()

    #   response = requests.post(
    #      URL,
    #      json={'content': prompt, 'histories': st.session_state.messages}
    #   )

    #   answer = response.json()['answer']

    #   if (response.status_code == 200):
    #     message_placeholder.markdown(answer.replace("\n", "<br>"), unsafe_allow_html=True)
    #   else:
    #     message_placeholder.markdown(f'응답 실패({response.status_code})')

    if (response.status_code == 200):
      st.session_state.messages.append({"role":"user", 'parts' : [prompt]})
      st.session_state.messages.append({'role': 'model', 'parts': [answer]})
      print(f'hmhm asdd messge {answer}')


# # %%
# import re
# text = """ddd ddd
# package=
# """

# def extract_package(text):
#   match = re.search(r'(.*)(package=\S*)', text)
#   if match:
#     result = text.split(r'package=')[0].rstrip('\n')
#     extracted = match.group(2).split('=')[1]
#   else:
#     result = text
#     extracted = None
#   return result, extracted

# aaa, bbb = extract_package(text)
# print(aaa)
# print(bbb)

# result = re.sub(r"package=.*$", "", text)
# print('---')
# print(result.rstrip('\n'))
# print('---')
# %%
