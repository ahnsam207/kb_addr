import os
import streamlit as st
import uuid
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.output_parsers import StrOutputParser


__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules['pysqlite3']
from langchain_chroma import Chroma

# os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
openai_api_key = os.environ["OPENAI_API_KEY"]
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

loader = PyPDFLoader("250304_kb_addr.pdf")
pages = loader.load_and_split()

# chat_history = []

text_splitter = RecursiveCharacterTextSplitter()#chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(pages)
persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(
    split_docs,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()
#Define the contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, fomulate a standalone question \
which can be understood without the chat history. Do Not answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_system_prompt ="""
You are an assistant for question-answering task. \
Use the following pieces of retrieved context to answer the question. \
If you don't konw the answer, just say that you don't konw. \
Keep the answer perfect. please use imogi with the answer.
Please answer in Korean and use respectful language.
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
llm = ChatOpenAI(model ="gpt-4o-mini")
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = StreamlitChatMessageHistory(key="chat_messages")

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key= "input",
    history_messages_key= "chat_history",
    output_messages_key= "answer",
)

# rag_chain = (
#     {"context":retriever | format_docs, "input":RunnablePassthrough()}
#     | qa_prompt
#     | llm
#     | StrOutputParser()
# )
st.header("KB 비상연락망 BOT")
st.write("KB 비상연락망 AI입니다. 무엇이든 물어보세요!")

for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input("질문을 입력해 주세요 :"):
    st.chat_message("human").write(prompt)
    st.session_state.messages.append({"role":"user", "content":prompt})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            config = {"configurable":{"session_id":"any"}}
            response = conversational_rag_chain.invoke({"input":prompt}, config)
            answer = response['answer']
            st.write(answer)
