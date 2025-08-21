from langgraph.graph import StateGraph, START, END 
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Annotated
from langchain_core.messages import HumanMessage , AnyMessage , AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
import os 

os.environ['LANGSMITH_PROJECT'] = 'ReAct Chatbot'

load_dotenv()
model = ChatOpenAI(model_name='gpt-4-turbo-2024-04-09')
search_tool = TavilySearchResults()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())


tool_model = model.bind_tools([search_tool,wikipedia,arxiv])

class State(BaseModel):
    messages : Annotated[list[AnyMessage], add_messages]

def bot(state : State):
    response = tool_model.invoke(state.messages)
    return {'messages' : [response]}

graph = StateGraph(State)

graph.add_node('chatbot' ,bot)
graph.add_node('tools' , ToolNode(tools = [search_tool,wikipedia,arxiv]))

graph.add_edge(START , 'chatbot')
graph.add_conditional_edges('chatbot' , tools_condition)
graph.add_edge('tools', 'chatbot')
graph.add_edge('chatbot' , END)

memory = MemorySaver()

workflow  = graph.compile(checkpointer = memory)

config= {'configurable' : {'thread_id' : 'thread_1'}}


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("<h1 style='text-align: center;'>ReAct Chatbot</h1>",unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything"):
    st.chat_message("user").markdown(prompt)

    st.session_state.history.append(HumanMessage(content = prompt))
    st.session_state.messages.append({"role": "user", "content": prompt})

    full_text = ''
    placeholder = st.empty()
    first_chunk = True

    placeholder.markdown("Thinking...")

    for message_chunk , metadata in workflow.stream({'messages' :st.session_state.history}, config=config,stream_mode='messages'):  
        if isinstance(message_chunk, AIMessage) and message_chunk.content:
            if first_chunk:
                    full_text = message_chunk.content
                    placeholder.markdown(full_text.rstrip("\n"))
                    first_chunk = False
            else:
                    full_text += message_chunk.content
                    placeholder.markdown(full_text.rstrip("\n"))


    st.session_state.history.append(AIMessage(content = full_text))
    st.session_state.messages.append({"role": "assistant", "content": full_text})



