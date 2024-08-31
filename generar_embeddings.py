from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tqdm
import time
import configparser
import getopt, sys


HUGGINGFACEHUB_API_TOKEN = ""
GOOGLE_API_KEY = ""
MODEL_EMBEDDINGS_OLLAMA = ""
MODEL_EMBEDDINGS_GOOGLE = ""
OUTPUT_PATH = ""
FAISS_GOOGLE_PATH = ""
FAISS_OLLAMA_PATH = ""
PDFS_PATH = ""
NUMEXPR_MAX_THREADS = ""


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


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
    
    
def generate_raw_file()
    for filename in os.listdir(PDFS_PATH + '/'):
    if filename.endswith(".pdf"):
        raw_text += get_pdf_text(PDFS_PATH + '/' + filename)
    with open(PDFS_PATH + '/rawdata.txt', 'w', encoding='utf-8') as archivo:
    archivo.write(raw_text)
    
    
def chunker_google(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def get_vector_store_google(text_chunks):
    print( len(text_chunks) )
    embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS_GOOGLE)

    start = 0
    end = 0
    index = 0
    for chunks in tqdm.tqdm(chunker_google(text_chunks, 180)):
        print( chunks[0][:10] )
        start = time.time()
        if index == 0:
            vector_store = FAISS.from_texts(chunks, embedding = embeddings)
        else:
            vector_store.add_texts(chunks)
            index += 1
        end = time.time()
        print( end-start )
        time.sleep( (60-end+start) if (end-start<60) else 0 ) # wait to not exceed any rate limits
    
    vector_store.save_local(FAISS_GOOGLE_PATH)
    

def get_vector_store_ollama(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_EMBEDDINGS_OLLAMA)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_OLLAMA_PATH)
    
    
    
def main():
    print("Starting...")
    load_config()
    
    # parametros de entrada por linea de comandos
    argumentList = sys.argv[1:]
    options = "hgm:"
    long_options = ["help", "gen_raw", "mode="]
    
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        gen_raw_file = False
        mode = "0"  ## 0 is Ollama; 1 is Google
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-g", "--gen_raw"):
                gen_raw_file = True
            elif currentArgument in ("-m", "--mode") and currentValue in ("0", "1"):
                mode = currentValue
            elif currentArgument in ("-h", "--help"):
                print("""
Uso: generar_embeddings [opciones]
  opciones:
    -h, --help: muestra esta ayuda.
    -g, --gen_raw: generate el archivo rawdata.txt desde cero con los archivos PDFs en la carpeta. Default no lo genera.
    -m, --mode: 0 para usar el modo Ollama, 1 para el modo Google. Default es 0.
""")
            
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))
    
    print("Working in {0} mode.".format("Ollama" if mode = "0" else "Google"))
    
    if ( gen_raw_file ):
        print("Generando rawdata file...")
        generate_raw_file()
        print("Generando rawdata file... OK")
    else:
        print("No se genera el rawdata file.")
        
    print("Generando los chunks...")
    text_chunks = ""
    with open(PDFS_PATH + '/rawdata.txt', 'r', encoding='utf-8') as archivo:
        raw_text1 = archivo.read()
        text_chunks = get_text_chunks(raw_text1)
    print("Generando los chunks... OK")
    
    print("Generando los FAISS...")
    if (mode = "0"):
        get_vector_store_ollama(text_chunks)
    else:
        get_vector_store_google(text_chunks)
    print("Generando los FAISS... OK")
    
    print("Terminado.")
    
    
if __name__ == "__main__":
   main()