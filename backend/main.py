from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

### FastAPI setup ###
app = FastAPI()

# Enable CORS for development (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatBot:

    def __init__(self):
        # Load model
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv('OPENAI_API_KEY'))

        # Load and split docs
        loader = TextLoader("./biography.txt", encoding='utf-8')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Setup vectorstore
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv('OPENAI_API_KEY'))
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function,
                                            persist_directory="./chroma_db", collection_name='v_db')
        retriever = vectorstore.as_retriever()

        # Prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and the latest user question 
                which might reference context in the chat history, formulate a standalone question 
                which can be understood without the chat history. Do NOT answer the question, 
                just reformulate it if needed and otherwise return it as is."""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are answering questions about a person named Christian. Use the provided 
                context to answer these questions. If the context does not provide an answer, just 
                say that Christian did not provide you with the necessary information to answer that question.
                Do not answer any questions that are not relevant to Christian.

                {context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        self.qa_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)

        # History store
        self.store = {}

        def get_session_history(session_id: str, store=self.store) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

# Create chatbot instance
bot = ChatBot()

@app.post("/chat")
def chat(request: ChatRequest):
    result = bot.conversational_rag_chain.invoke(
        {"input": request.query},
        config={"configurable": {"session_id": request.session_id}}
    )
    return {"response": result["answer"]}
