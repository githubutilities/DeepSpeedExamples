import os
import sys
import json

import requests
from transformers import AutoModel, AutoTokenizer
import streamlit as st
from streamlit_chat import message


def get_model_meta():
    with open(os.path.expanduser('~/chatgpt_route.json')) as f:
        model_meta = json.loads(f.read())
    return model_meta
model_str2meta = get_model_meta()
MAX_TURNS = 10
MAX_BOXES = MAX_TURNS * 2

def generate(model_name=None, url=None, datas={
        'query_list': [
            {'query':'Human: 1955年美国总统是谁?', 'history': []},
            {'query':'Human: 5*1000=多少?', 'history': []},
        ]
    }, debug=False):
    if url is None:
        url = model_meta[model_name]['url']
    data_json = json.dumps(datas)
    resp = requests.post(url=url, data=data_json)
    ret = resp.json()
    if debug:
        print(ret)
    resp_list = ret.pop('result', [])
    answer_list = [e.get('answer', '') for e in resp_list]
    return answer_list[0]

def predict(input, model_str, mode='left', container=None, **kwargs):
    if '{}_history'.format(mode) not in st.session_state:
        st.session_state['{}_history'.format(mode)] = []
    history = st.session_state['{}_history'.format(mode)]

    if container is None:
        container = st.container()

    #FIXME: echo
    input_response = input + 'b'
    datas = {
        'query_list': [
            {'query': input, 'history': []},
        ]
    }
    input_response = generate(url=model_str2meta[model_str]['url'], datas=datas)

    # 处理历史
    with container:
        if len(history) > 0:
            if len(history)>MAX_BOXES:
                history = history[-MAX_TURNS:]
            for i, (query, response) in enumerate(history):
                message(query, avatar_style="big-smile", key=f"{mode}_{i}_user")
                message(response, avatar_style="bottts", key=f"{mode}_{i}")

        message(input, avatar_style="big-smile", key=f"{mode}_{len(history)}_user")
        message(input_response, avatar_style="bottts", key=f"{mode}_{len(history)}")
        st.write("AI正在回复:")
        with st.empty():
            history.append((input, input_response))
            for c in input_response:
                st.write(c)
    return history

st.set_page_config(
    page_title="Chat演示白板",
    page_icon=":robot:"
)

left_container, right_container = st.columns(2)
# create a prompt text for the text generation
prompt_text = st.text_area(label="用户命令输入",
            height = 100,
            placeholder="请在这儿输入您的命令")
#model_str2meta = {
#    'ChatGLM原始': 'abcd',
#}
left_model_str = st.sidebar.selectbox(label='左侧模型', options=list(model_str2meta.keys()) + ['none'])
right_model_str = st.sidebar.selectbox(label='右侧模型', options=list(model_str2meta.keys()) + ['none'])
 
#max_length = st.sidebar.slider(
#    'max_length', 0, 4096, 2048, step=1
#)
#top_p = st.sidebar.slider(
#    'top_p', 0.0, 1.0, 0.6, step=0.01
#)
#temperature = st.sidebar.slider(
#    'temperature', 0.0, 1.0, 0.95, step=0.01
#)

kwargs = {}
if st.button("发送", key="predict"):
    with st.spinner("AI正在思考，请稍等........"):
        # text generation
        if left_model_str != 'none':
            st.session_state["left_history"] = predict(prompt_text, model_str=left_model_str, mode='left',   container=left_container, **kwargs)
        if right_model_str != 'none':
            st.session_state["right_history"] = predict(prompt_text, model_str=right_model_str, mode='right', container=right_container, **kwargs)

