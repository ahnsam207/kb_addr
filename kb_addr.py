from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']
from langchain_chroma import Chroma

os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loader = PyPDFLoader(r"250304_kb_addr.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(pages)
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    split_docs,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()

qa_system_prompt ="""
You are an assistant for question-answering task. \
Use the following pieces of retrieved context to answer the question. \
If you don't konw the answer, just say that you don't konw. \
Keep the answer perfect. please use imogi with the answer.
Please answer in Korean and use respectful language.
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model ="gpt-4o-mini")
rag_chain = (
    {"context":retriever | format_docs, "input":RunnablePassthrough()}
    | qa_prompt
    | llm
    | StrOutputParser()
)
st.header("KB 비상연락망 BOT")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"KB 비상연락망 AI입니다.  AI도 실수 할 수 있어요.^^"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input("질문을 입력해 주세요 :"):
    st.chat_message("human").write(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt)
            st.session_state.messages.append({"role":"assistant", "content":response})
            st.write(response)
