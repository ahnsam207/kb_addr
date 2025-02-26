import streamlit as st
import requests
import base64
# 랭체인 활용시 참고할 만한 프롬프트가 다수 존재하며 공유와 관리를 위한 랭체인 관리 플랫폼
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI


# GitHub 정보
GITHUB_REPO = "ahnsam207/kb_addr"    ######## 폴더 지정 해야함 #######
GITHUB_TOKEN =  st.secrets["git_token"]   # 개인 액세스 토큰 입력
BRANCH = "main"  # 사용할 브랜치

# 파일 업로드

st.title("KB 비상연락망 DB 생성")

# 헌법 PDF 파일 로드
loader = PyPDFLoader("250304_kb_addr.pdf")
pages = loader.load_and_split()
# PDF 파일을 1000자 청크로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(pages)

#ChromaDB에 청크들을 벡터 임베딩으로 저장(OpenAI 임베딩 모델 활용)
vector_store = Chroma.from_documents(docs, persist_directory="./chroma_db", model="text-embedding-3-small")
retriever = vector_store.as_retriever()
