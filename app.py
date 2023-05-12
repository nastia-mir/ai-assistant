import streamlit as st
import chromadb
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from decouple import Config, RepositoryEnv

env = Config(RepositoryEnv('.env'))
api_key = env.get('OPENAI_API_KEY')

# embeddings = OpenAIEmbeddings(openai_api_key=api_key)
# vectorstore = Chroma(embedding_function=embeddings)
# retriever = vectorstore.as_retriever(search_kwargs=dict(k=1))
# memory = VectorStoreRetrieverMemory(retriever=retriever)


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'input' not in st.session_state:
    st.session_state['input'] = ""
if 'stored_session' not in st.session_state:
    st.session_state['stored_session'] = []


def get_text():
    input_text = st.text_input('You:', st.session_state['input'], key='input',
                               placeholder='Hi! I am your AI assistant. Ask me anything.',
                               label_visibility='hidden')
    return input_text


def new_chat():
    save = []
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        save.append('User:' + st.session_state['past'][i])
        save.append('Bot:' + st.session_state['generated'][i])
        # memory.save_context({'input': st.session_state['past'][i]}, {'output': st.session_state['generated'][i]})
    st.session_state['stored_session'].append(save)
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['input'] = ""
    st.session_state.entity_memory.entity_store = {}
    st.session_state.entity_memory.buffer.clear()


st.title("AI assistant")
st.button('New chat', on_click=new_chat)


llm = ChatOpenAI(
    temperature=0,
    openai_api_key=api_key,
    model_name='gpt-3.5-turbo'
)

if 'entity_memory' not in st.session_state:
    st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k=100)


Conversation = ConversationChain(
        llm=llm,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=st.session_state.entity_memory
    )

user_input = get_text()

if user_input:
    output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

with st.expander("Conversation"):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.success(st.session_state['generated'][i])
        st.info(st.session_state['past'][i])