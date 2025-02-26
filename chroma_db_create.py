import streamlit as st
import requests
import base64
from langchain.document_loaders import PyPDFLoader  # 최신 경로로 수정
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 모듈명 수정
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma  # Chroma 경로 수정
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# GitHub 정보
GITHUB_REPO = "ahnsam207/kb_addr"
GITHUB_TOKEN = st.secrets["git_token"]  # 개인 액세스 토큰
BRANCH = "main"

# 파일 업로드
st.title("KB 비상연락망 DB 생성")

# PDF 파일 로드
loader = PyPDFLoader("250304_kb_addr.pdf")
pages = loader.load_and_split()

# PDF 파일을 1000자 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

# ChromaDB에 청크들을 벡터 임베딩으로 저장 (OpenAI 임베딩 모델 활용)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")  # model 인자 이동
vector_store = Chroma.from_documents(docs, embedding_function=embedding_function, persist_directory="./chroma_db")
vector_store.persist()  # ChromaDB 저장

retriever = vector_store.as_retriever()
