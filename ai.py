import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage

llm = AzureChatOpenAI(
    temperature=0.3,
    deployment_name="gpt-35-turbo-16k",
    azure_endpoint="https://gmk.openai.azure.com/",
    openai_api_type="azure",
    api_version="2023-07-01-preview",
    api_key="1a85d37104f5410aafc67aca6a8837df",
    streaming=False,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, 'db')
embeddings = FastEmbedEmbeddings()
messages = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

retriever_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    ('user',"{input}"),
    ("assistant",'Based on the chat history and question, what should I search for?')
])
retriever_chain = create_history_aware_retriever(
    llm,
    retriever,
    prompt=retriever_prompt
)

response_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}"),
    ("assistant", """
    Based on the following context and chat history, answer the question:
    
    Context: {context}
    
    If you don't find the information in the context, say so.
    """)
])

combine_docs_chain = create_stuff_documents_chain(
    llm,
    response_prompt
)

retrieval_chain = create_retrieval_chain(
    retriever_chain,
    combine_docs_chain
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

def embed_document(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified file not found : {path} ")
    
    msg = "Started process of embedding."
    print(msg)

    loader = TextLoader(path, encoding='utf-8')
    documents = loader.load()

    docs = text_splitter.split_documents(documents=documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    # print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("Creating vectorstore")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)

    msg = "Finished embedding"
    print(msg)

def query(query):
    if not query:
        return "Please provide a  query"
    messages.chat_memory.add_user_message(query)
    
    response = retrieval_chain.invoke({
        "chat_history":messages.chat_memory.messages,
        "input":query,
        "question":query
    })

    print("response from AI ",response)
    messages.chat_memory.add_ai_message(response["answer"])
    return response["answer"]