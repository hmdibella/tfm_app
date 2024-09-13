import streamlit as st
import warnings
import configparser
import os
import logging
import random
import string
import time
from langchain_core.chat_history import BaseChatMessageHistory
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
#from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory



GOOGLE_API_KEY = ""
MODEL_EMBEDDINGS_GOOGLE = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_LLM_GOOGLE = "gemini-1.5-pro"
OUTPUT_PATH = "output"
FAISS_GOOGLE_PATH = "output/faiss_index"
NUMEXPR_MAX_THREADS = "16"
K_VALUE = "8"

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Dado un historial de chat y la última pregunta del usuario que podría referenciar contexto en dicho historial, reformula la pregunta para que sea independiente y comprensible sin necesidad de acceder al historial. No respondas la pregunta; solo reformúlala si es necesario, o devuélvela tal como está si ya es clara.
"""

PROMPT_TEMPLATE_GOOGLE = """Eres un asistente inteligente. Responde a la pregunta basándote únicamente en la información 
proporcionada en el contexto, específico para España.
Siempre prioriza mencionar artículos o leyes que estén directamente relacionados con la pregunta del usuario.
Si el contexto no contiene la respuesta, responde diciendo que la respuesta no está disponible en el contexto proporcionado, pidiendo disculpas, aclarando que solo puedes responder sobre el Código de Tráfico y Seguridad Vial de España. Intenta buscar la respuesta en el buscador de Google en este caso y dar una respuesya basado en la búsqueda y el mejor resultado de la misma.
Si la pregunta se refiere a un tema legal, intenta identificar y citar el artículo de la ley que aplique.
Genera las respuestas con tus propias palabras, sin copiar directamente el contenido, y asegúrate de que sean claras, fáciles de entender y aplicables a un público juvenil.
Por ejemplo, si la respuesta incluyera el Artículo 48, responde con su título así "Artículo 48. Velocidades máximas en vías fuera de poblado." seguido del código del BOE correspondiente, dentro de la explicación.
Siempre aclara que tus respuestas no deben ser tomadas para toma de decisiones y se debe consultar siempre con un experto.
Finalmente, pregúntale al usuario si has respondido la pregunta. En caso que la respuesta sea afirmativa, agradécele por confirmar; en caso que la respuesta sea negativa, intenta explicarlo con otras palabras e intentando explayarte un poco más, o intentando resumir la respuesta si esta ha sido extensa.

Contexto:
{context}
"""


def load_config():
    global GOOGLE_API_KEY
    global MODEL_EMBEDDINGS_GOOGLE
    global OUTPUT_PATH
    global FAISS_GOOGLE_PATH
    global LOG_PATH
    global NUMEXPR_MAX_THREADS
    global K_VALUE
    
    config = configparser.ConfigParser()
    config.read('streamlit_google_history_final.ini')
    GOOGLE_API_KEY = config['KEYS']['google_api_key']
    MODEL_EMBEDDINGS_GOOGLE = config['MODELS']['model_embeddings_google']
    MODEL_LLM_GOOGLE = config['MODELS']['model_llm_google']
    OUTPUT_PATH = config['DEFAULT']['output_path']
    FAISS_GOOGLE_PATH = config['DEFAULT']['faiss_google_path']
    NUMEXPR_MAX_THREADS = config['DEFAULT']['numexpr_max_threads']
    LOG_PATH = config['DEFAULT']['log_path']
    K_VALUE = config['MODELS']['k_value']


def get_conversational_chain(retriever, session_id):
    
    llm = ChatGoogleGenerativeAI(model = MODEL_LLM_GOOGLE, temperature = 0.3)
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT_TEMPLATE_GOOGLE),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    msgs = StreamlitChatMessageHistory(key=session_id)
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    return conversational_rag_chain


def user_input(user_question, session_id):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDINGS_GOOGLE)
    new_db = FAISS.load_local(FAISS_GOOGLE_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = new_db.as_retriever(search_kwargs={"k": int(K_VALUE)})
    chain = get_conversational_chain(retriever, session_id)
    response = chain.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": session_id}
        },
    )
    return response
    
    
def response_generator(prompt, session_id, logger):
    start_model_exec = time.time()
    response = user_input(prompt, session_id)
    end_model_exec = time.time()
    resp_text = "{0} *(Tiempo de respuesta: {1:.2f} seg.)*.".format(response["answer"], end_model_exec - start_model_exec)
    logger.debug(session_id + ' - Contexto ' + str(response) )
    for word in resp_text.split(" "):
        yield word + " "
        time.sleep(0.05)


def response_generator_bot(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)
        
        
def create_new_chat():
    st.session_state["chats_count"] += 1
    st.session_state["history_chats"] = st.session_state["history_chats"] + ["Chat " + str(st.session_state["chats_count"])]
    st.session_state["current_chat_index"] = st.session_state["chats_count"]-1


def main():

    st.set_page_config(page_title="Semaforín v1.0", page_icon="🤖", layout="wide")
    st.markdown("""
<style>
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>""",
    unsafe_allow_html=True,
)
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    load_config()
    os.environ["NUMEXPR_MAX_THREADS"] = NUMEXPR_MAX_THREADS
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

    logger = logging.getLogger(__name__)
    log_ruta = LOG_PATH + '/streamlit_google_history_final.log'
    logging.basicConfig(filename=log_ruta, level=logging.DEBUG, 
    format='%(asctime)s.%(msecs)03d %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    # Initialization
    if "current_chat_index" not in st.session_state:
        st.session_state["current_chat_index"] = 0
        st.session_state["chats_count"] = 1
        st.session_state["history_chats"] = ["Chat 1"]
    
    session = "session_id" + str(st.session_state["current_chat_index"])
    messages = "messages" + str(st.session_state["current_chat_index"])
    if session not in st.session_state:
        st.session_state[session] = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        st.session_state[messages] = []
        st.session_state[messages].append({"role": "Semaforín", "content": "Hola! 👋 Soy Semaforín 🤖, tu colega-bot que responde preguntas ❓ sobre las leyes de Tráfico y Seguridad Vial 🚗 en España. Hazme las preguntas y yo trataré de responderlas 💪."})
        
  
    with st.sidebar:
        st.title("Semaforín v1.0")
        st.subheader( "Bienvenida/o a Semaforín v1.0! El primer colega-bot que responde sobre las leyes del Código de Tráfico y Seguridad Vial de España." )
        current_chat = st.radio(
            label="Lista de chats",
            options=st.session_state["history_chats"],
            #index=st.session_state["current_chat_index"],
            key="chat_radiobutton"
        )
        if current_chat:
            st.session_state["current_chat_index"] = st.session_state["history_chats"].index(current_chat)
                
        create_chat_button = st.button("Nuevo chat", use_container_width=True, key="create_chat_button")
        if create_chat_button:
            create_new_chat()
            st.rerun()
    
    avatar_assistant = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_bot.jpg'
    avatar_user = 'https://raw.githubusercontent.com/hmdibella/tfm_app/main/perfil_humano.jpg'
    
    # Display chat messages from history on app rerun
    current_state = st.session_state["messages"+str(st.session_state["current_chat_index"])]
    for message in current_state:
        avatar_img = avatar_assistant if message["role"] == "Semaforín" else avatar_user
        with st.chat_message(message["role"], avatar=avatar_img):
            st.markdown(message["content"])
            
  
    # Accept user input
    if prompt := st.chat_input("Haz la pregunta aquí"):
        # Add user message to chat history
        current_state.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user", avatar=avatar_user):
            st.markdown(prompt)
            
        session = 'session_id'+str(st.session_state["current_chat_index"])
        logger.info(st.session_state[session] + ' - Pregunta nueva: ' + prompt )

        with st.spinner("Procesando, por favor espere..."):
            
            # Display assistant response in chat message container
            with st.chat_message("Semaforín", avatar=avatar_assistant):
                response = st.write_stream(response_generator(prompt, st.session_state[session], logger))

            # Add assistant response to chat history
            current_state.append({"role": "Semaforín", "content": response})
            logger.info(st.session_state[session] + ' - Respuesta: ' + response )

  
if __name__ == "__main__":
   main()
