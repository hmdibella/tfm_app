import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
import time
import logging
import configparser
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)


HUGGINGFACEHUB_API_TOKEN = ""
GOOGLE_API_KEY = ""
MODEL_EMBEDDINGS_OLLAMA = "paraphrase-xlm-r-multilingual-v1"
MODEL_EMBEDDINGS_GOOGLE = "models/embedding-001"
OUTPUT_PATH = "output"
FAISS_GOOGLE_PATH = "output/faiss_index_google"
FAISS_OLLAMA_PATH = "output/faiss_index"
NUMEXPR_MAX_THREADS = "16"


def load_config():
    global HUGGINGFACEHUB_API_TOKEN
    global GOOGLE_API_KEY
    global MODEL_EMBEDDINGS_OLLAMA
    global MODEL_EMBEDDINGS_GOOGLE
    global OUTPUT_PATH
    global FAISS_GOOGLE_PATH
    global FAISS_OLLAMA_PATH
    global LOG_PATH
    global NUMEXPR_MAX_THREADS
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    HUGGINGFACEHUB_API_TOKEN = config['KEYS']['huggingfacehub_api_token']
    GOOGLE_API_KEY = config['KEYS']['google_api_key']
    MODEL_EMBEDDINGS_OLLAMA = config['MODELS']['model_embeddings_ollama']
    MODEL_EMBEDDINGS_GOOGLE = config['MODELS']['model_embeddings_google']
    OUTPUT_PATH = config['DEFAULT']['output_path']
    FAISS_GOOGLE_PATH = config['DEFAULT']['faiss_google_path']
    FAISS_OLLAMA_PATH = config['DEFAULT']['faiss_ollama_path']
    NUMEXPR_MAX_THREADS = config['DEFAULT']['numexpr_max_threads']
    LOG_PATH = config['DEFAULT']['log_path']


def get_conversational_chain_ollama(model_name):
    prompt_template_llama = """
You are an intelligent assistant. Answer the question based solely on the information provided in the context, specific for Spain. Also consider the previous conversations you had with the user, to answer to the question.
Do not add any information beyond what is given. 
If the context does not contain the answer, respond with: "The answer is not available in the provided context."
Be concise and accurate. All the answers must be in Spanish language from Spain, in an informal manner.
Answers must be clear, precise, and unambiguous.
Give the answers like the user is not a lawyer or an expert in the legal area.
Explicitly avoid phrases such as "según el documento", "según el capítulo", "en el texto", "como se menciona en el artículo", or any implication of external texts. Do not construct questions that require knowledge of the structure of the document or the location of information in it.
Include the content-specific information that supports the answer to allow the answer to be independent of any external text.
If the content lacks sufficient information to form a complete answer, do not force one.
Create the answers in your own words; Direct copying of content is not permitted.
NEVER mention the words "documento", "texto", "presentación", "archivo", "tabla", "artículo", "ley", "capítulo", "preámbulo", "título preliminar", "disposición" or "disposiciones generales" in your questions or answers.
ALWAYS make sure that all answers are accurate, self-contained, and relevant, without relying on any original document or text or implying its existence, strictly avoiding any invention or speculation.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt_template_gemma = """<start_of_turn>user
You are an intelligent assistant. Answer the question based solely on the information provided in the context, specific for Spain. 
Do not add any information beyond what is given. 
If the context does not contain the answer, respond with: "The answer is not available in the provided context."
Be concise and accurate. All the answers must be in Spanish language from Spain, in an informal manner.
Answers must be clear, precise, and unambiguous.
Give the answers like I was not a lawyer or an expert in the legal area.
Explicitly avoid phrases such as "según el documento", "según el capítulo", "en el texto", "como se menciona en el artículo", or any implication of external texts. Do not construct questions that require knowledge of the structure of the document or the location of information in it.
Include the content-specific information that supports the answer to allow the answer to be independent of any external text.
If the content lacks sufficient information to form a complete answer, do not force one.
Create the answers in your own words; Direct copying of content is not permitted.
NEVER mention the words "documento", "texto", "presentación", "archivo", "tabla", "artículo", "ley", "capítulo", "preámbulo", "título preliminar", "disposición" or "disposiciones generales" in your questions or answers.
ALWAYS make sure that all answers are accurate, self-contained, and relevant, without relying on any original document or text or implying its existence, strictly avoiding any invention or speculation.

Context:
{context}

Question:
{question}

<start_of_turn>model
"""
    # Usar Ollama como LLM
    model = Ollama(model=model_name)
    
    prompt_template = prompt_template_llama if model_name.startswith('llama') else prompt_template_gemma
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "prev_conv"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain
    
    
def get_conversational_chain_google():
    prompt_template = """
    You are an intelligent assistant. Answer the question based solely on the information provided in the context, specific for Spain. 
Do not add any information beyond what is given. 
If the context does not contain the answer, respond with: "The answer is not available in the provided context."
Be concise and accurate. All the answers must be in Spanish language from Spain, in an informal manner. Answers must be clear, precise, and unambiguous.
Give the answers as such the user would not be a lawyer or an expert in the legal area.
Explicitly avoid phrases such as "según el documento", "según el capítulo", "en el texto", "como se menciona en el artículo", or any implication of external texts. Do not construct questions that require knowledge of the structure of the document or the location of information in it.
Include the content-specific information that supports the answer to allow the answer to be independent of any external text.
Create the answers in your own words; Direct copying of content is not permitted.
NEVER mention the words "documento", "texto", "presentación", "archivo", "tabla", "artículo", "ley", "capítulo", "preámbulo", "título preliminar", "disposición" or "disposiciones generales" in your questions or answers.
ALWAYS make sure that all answers are accurate, self-contained, and relevant, without relying on any original document or text or implying its existence, strictly avoiding any invention or speculation.

Context:
{context}
{prev_conv}


Question:
{question}
    

Answer:
"""
    model = ChatGoogleGenerativeAI(model = "gemini-pro",temperature = 0.3)
    prompt = PromptTemplate(template= prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type = "stuff", prompt = prompt)
    return chain


def user_input_ollama(user_question, model_name1, prev_conv, chunk_size, k_value):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDINGS_OLLAMA)
    new_db = FAISS.load_local(FAISS_OLLAMA_PATH + "_" + chunk_size, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=int(k_value))
    chain = get_conversational_chain_ollama(model_name1)
    response = chain.run({"context": docs, "question": user_question, "prev_conv": prev_conv})
    return response
    
    
def user_input_google(user_question, prev_conv, chunk_size, k_value):
    embeddings = GoogleGenerativeAIEmbeddings(model = MODEL_EMBEDDINGS_GOOGLE)
    new_db = FAISS.load_local(FAISS_GOOGLE_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain_google()
    response = chain(
        {"input_documents":docs, "question": user_question, "prev_conv": prev_conv}, return_only_outputs=True)
    return response["output_text"]
    
    
def response_generator(prompt, model_name, prev_conv, chunk_size, k_value):
    start_model_exec = time.time()
    if model_name.startswith('Gemini'):
        response = user_input_google(prompt, prev_conv, chunk_size, k_value)
    else:
        response = user_input_ollama(prompt, model_name, prev_conv, chunk_size, k_value)
    end_model_exec = time.time()
    resp_text = "{0} - {3} - {4}: {1}\n\n(Tiempo de respuesta: {2:.2f} seg.).".format(model_name, response, end_model_exec - start_model_exec, chunk_size, k_value)
    for word in resp_text.split():
        yield word + " "
        time.sleep(0.05)


def main():
    
    load_config()
    os.environ["NUMEXPR_MAX_THREADS"] = NUMEXPR_MAX_THREADS
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    #genai.configure(api_key = GOOGLE_API_KEY)

    logger = logging.getLogger(__name__)
    log_ruta = LOG_PATH + '/streamlit_prueba.log'
    logging.basicConfig(filename=log_ruta, level=logging.INFO, 
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
  
    st.set_page_config(page_title="Chat legal v1.0", page_icon="🇪🇸")  # Set title and icon
  
    model_chat = st.sidebar.selectbox( 'Elije un modelo', [
        'llama3.1:8b-instruct-q4_K_M', 
        'llama3.1:8b-instruct-fp16', 
        'llama3.1:8b-instruct-q8_0', 
        'llama3:8b-instruct-fp16',
        'llama3:8b-instruct-q4_K_M',
        'llama3:8b-instruct-q8_0',
        'gemma2:9b-instruct-fp16',
        'gemma2:9b-instruct-q4_K_M',
        'gemma2:9b-instruct-q8_0',
        'Gemini-pro'],
        index=0 )
        
    chunk_size = st.sidebar.selectbox( 'Elije un chunk size', [
        '10000', 
        '5000', 
        '2000',
        '1000'],
        index=0 )
        
    k_value = st.sidebar.number_input( "Inserta el valor de k", value=1, placeholder="Ingresa un entero...", min_value=1, format="%d")
    
    st.header("Chat con el modelo", divider="gray")
  
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Inicializar el contexto de la conversación
    if "context" not in st.session_state:
        st.session_state.context = ""
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
  
    # Accept user input
    if prompt := st.chat_input("Haz una pregunta sobre las leyes de consumo de España"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        logger.info('Pregunta nueva: ' + prompt )

        with st.spinner("Procesando, por favor espere..."):
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                #response = st.write_stream(response_generator(prompt, model_chat, st.session_state.context, chunk_size, k_value))
                response = st.write_stream(response_generator(prompt, model_chat, "", chunk_size, k_value))

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            logger.info('Respuesta lista. Modelo usado: ' + model_chat )
            logger.info('Respuesta: ' + response )
            
            # Agregar la pregunta y la respuesta al contexto
            st.session_state.context += f"Pregunta: {prompt}\n"
            st.session_state.context += f"Respuesta: {response}\n"

  
if __name__ == "__main__":
   main()
