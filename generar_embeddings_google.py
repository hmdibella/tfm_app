import warnings
import getopt, sys
import configparser
import subprocess
import os
import tqdm
import time
import xml.etree.ElementTree as ET
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


from langchain_huggingface import HuggingFaceEmbeddings


warnings.filterwarnings("ignore", category=FutureWarning)


GOOGLE_API_KEY = ""
MODEL_EMBEDDINGS_GOOGLE = "models/embedding-001"
OUTPUT_PATH = "output"
FAISS_GOOGLE_PATH = "output/faiss_index"
NUMEXPR_MAX_THREADS = "16"


BOE_CODES_CONSUMO = ["BOE-A-1978-31229", "BOE-A-2006-22950", "BOE-A-2007-20555", "BOE-A-1998-8789", "BOE-A-1999-24356", "BOE-A-1996-1072", "BOE-A-2004-21421", "BOE-A-1991-628", "BOE-A-1988-26156", "BOE-A-2013-12888", "BOE-A-2012-15595", "BOE-A-2000-323", "BOE-A-2017-12659", "BOE-A-2020-14046", "BOE-A-2003-23646", "BOE-A-2002-13758", "BOE-A-2024-15208", "BOE-A-1995-25444", "BOE-A-1983-19755", "BOE-A-2007-22440", "BOE-A-2000-24118", "BOE-A-2000-16561", "BOE-A-2007-21491", "BOE-A-2008-17629", "BOE-A-2004-511", "BOE-A-1990-14814", "BOE-A-2005-6795", "BOE-A-2020-15872", "BOE-A-2011-14252", "BOE-A-2018-6651", "BOE-A-2013-3210", "BOE-A-1988-28089", "BOE-A-1999-17996", "BOE-A-2015-2293", "BOE-A-1988-2809", "BOE-A-2009-18731", "BOE-A-1986-18896", "BOE-A-2007-22440", "BOE-A-2012-14696", "BOE-A-2023-24840", "BOE-A-2014-6726", "BOE-A-2015-1455", "BOE-A-2011-4117", "BOE-A-2012-9111", "BOE-A-1989-11181", "BOE-A-2011-2618", "BOE-A-2013-5073", "BOE-A-2009-5391", "BOE-A-2017-653", "BOE-A-2019-3108", "BOE-A-2019-3814", "BOE-A-2019-6299", "BOE-A-2011-10970", "BOE-A-1998-16717", "BOE-A-1954-15448", "BOE-A-2018-16036", "BOE-A-2017-13644", "BOE-A-2019-18425", "BOE-A-2019-4906", "BOE-A-2019-18677", "BOE-A-2019-3113", "BOE-A-2007-13411", "BOE-A-2012-3394", "BOE-A-2022-19403", "BOE-A-2022-19535", "BOE-A-2013-3199", "BOE-A-2011-17015", "BOE-A-2010-10315", "BOE-A-2015-11932", "BOE-A-2019-4955", "BOE-A-2012-9058", "BOE-A-2010-15521", "BOE-A-2020-7869", "BOE-A-2012-14363", "BOE-A-2004-5290", "BOE-A-2015-11429", "BOE-A-2023-14051", "BOE-A-2007-6115", "BOE-A-2022-11589", "BOE-A-2018-13593", "BOE-A-2017-11505", "BOE-A-2004-884", "BOJA-b-2012-90004", "BOJA-b-2012-90003", "BOE-A-2016-6309", "BOE-A-2007-3601", "BOE-A-2015-5328", "BOE-A-2005-18179", "BOE-A-2016-11670", "BOE-A-2014-8820", "BOE-A-2014-11805", "BOE-A-2003-4608", "BOE-A-2006-5810", "BOE-A-2002-6232", "BOE-A-2010-13115", "BOE-A-2017-11320", "BOE-A-2010-738", "BOE-A-2019-6772", "BOE-A-2010-11731", "BOE-A-2020-1534", "BOE-A-2015-3280", "BOCL-h-2014-90371", "BOE-A-1999-2772", "BOE-A-1998-20651", "BOE-A-1999-17589", "BOE-A-1997-18549", "DOGV-r-2019-90594", "BOE-A-2011-6875", "BOE-A-2017-2422", "BOE-A-2019-3492", "BOE-A-2002-11417", "BOE-A-2012-5595", "BOE-A-2004-21335", "BOE-A-2007-2207", "BOE-A-2011-1649", "BOE-A-2018-1751", "BOE-A-2013-7476", "BOE-A-2013-4464", "BOE-A-2005-5660", "BOE-A-1997-20318", "BOE-A-2023-470", "BOE-A-2001-15779", "BOE-A-1989-23886", "BOE-A-2003-10080", "BOE-A-2012-1540", "BOE-A-2019-3705", "BOE-A-2023-13537", "BOE-A-2003-912", "BOE-A-2011-2621", "BOE-A-1996-21850", "BOE-A-2007-9420"]

BOE_CODES_TRAFFIC = ["BOE-A-2003-23514", "BOE-A-2024-1773", "BOE-A-2024-4143", "BOE-A-2024-3943", "BOE-A-2024-5842", "BOE-A-2021-3984", "BOE-A-2021-4194", "BOE-A-2009-9481", "BOE-A-2003-19801", "BOE-A-2010-3471", "BOE-A-2021-6624", "BOE-A-2005-13723", "BOE-A-2024-17780", "BOE-A-1994-8985", "BOE-A-2010-18102", "BOE-A-2010-19282", "BOE-A-2007-10557", "BOE-A-2005-7137", "BOE-A-1979-23768", "BOE-A-2004-18911", "BOE-A-2008-14915", "BOE-A-1999-1826", "BOE-A-1995-19000", "BOE-A-1999-23609", "BOE-A-2010-9994", "BOE-A-2010-11154", "BOE-A-2017-12841", "BOE-A-2017-6512", "BOE-A-2008-14194", "BOE-A-2004-1739", "BOE-A-2018-14948", "BOE-A-2010-4180", "BOE-A-2021-3982", "BOE-A-1995-25444"]


TASK_TYPES = ["task_type_unspecified", "retrieval_query", "retrieval_document", "semantic_similarity", "classification", "clustering"]


def load_config():
    global GOOGLE_API_KEY
    global MODEL_EMBEDDINGS_GOOGLE
    global OUTPUT_PATH
    global FAISS_GOOGLE_PATH
    global LOG_PATH
    global NUMEXPR_MAX_THREADS
    global CHUNK_SIZE
    
    config = configparser.ConfigParser()
    config.read('streamlit_google_history_final.ini')
    GOOGLE_API_KEY = config['KEYS']['google_api_key']
    MODEL_EMBEDDINGS_GOOGLE = config['MODELS']['model_embeddings_google']
    OUTPUT_PATH = config['DEFAULT']['output_path']
    FAISS_GOOGLE_PATH = config['DEFAULT']['faiss_google_path']
    NUMEXPR_MAX_THREADS = config['DEFAULT']['numexpr_max_threads']
    LOG_PATH = config['DEFAULT']['log_path']


def get_pdf_text(pdf_docs):
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text
    
    
def get_text_chunks_boe():
    concatenated_texts = []
    for code in BOE_CODES_TRAFFIC:
        curl_command = [
            "curl", "-L", "-X", "GET", "-H", "Accept: application/xml",
            f"https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/{code}/texto"
        ]
        
        # Ejecutar el comando y capturar la salida
        result = subprocess.run(curl_command, stdout=subprocess.PIPE)

        # Obtener el contenido del XML como texto
        xml_content = result.stdout.decode('utf-8')

        # Parsear el contenido del XML
        root = ET.fromstring(xml_content)

        precepto_blocks = []

        for block in root.findall(".//bloque[@tipo='precepto']"):
            precepto_blocks.append(block)

        for precepto in precepto_blocks:
            article_content = precepto.find(".//p[@class='articulo']").text if precepto.find(".//p[@class='articulo']") is not None else ""
            # Extract paragraphs' text
            paragraph_contents = [p.text if p.text is not None else "" for p in precepto.findall(".//p[@class='parrafo']")]
            # Concatenate article content with paragraph contents
            full_text = article_content + " " + " ".join(paragraph_contents).strip()
            concatenated_texts.append(full_text)
            
    return concatenated_texts
    
    
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = math.ceil(CHUNK_SIZE*0.1))
    chunks = text_splitter.split_text(text)
    return chunks
    
    
def generate_raw_file():
    for filename in os.listdir(PDFS_PATH + '/'):
        if filename.endswith(".pdf"):
            raw_text += get_pdf_text(PDFS_PATH + '/' + filename)
    with open(PDFS_PATH + '/rawdata.txt', 'w', encoding='utf-8') as archivo:
        archivo.write(raw_text)
    return len(raw_text)
    
    
def chunker_google(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

'''
def get_vector_store(text_chunks):

    for task_type in TASK_TYPES:
        embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_EMBEDDINGS_GOOGLE, task_type=task_type)

        start = 0
        end = 0
        index = 0
        for chunks in tqdm.tqdm(chunker_google(text_chunks, 180)):
            print( chunks[0][:] )
            print( chunks[1][:] )
            start = time.time()
            if index == 0:
                vector_store = FAISS.from_texts(chunks, embedding = embeddings)
            else:
                vector_store.add_texts(chunks)
                index += 1
            end = time.time()
            print( end-start )
            time.sleep( (60-end+start) if (end-start<60) else 0 ) # wait to not exceed any rate limits
        
        vector_store.save_local(FAISS_GOOGLE_PATH + "_" + task_type)
'''
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-xlm-r-multilingual-v1")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("output/faiss_index_ollama")
    

def main():
    # parametros de entrada por linea de comandos
    argumentList = sys.argv[1:]
    options = "h"
    long_options = ["help"]
    
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        gen_raw_file = False
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--help"):
                print("""
Uso: generar_embeddings [opciones]
  opciones:
    -h, --help: muestra esta ayuda.
""")
                return
            
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))
    
    print("Comenzando...")
    load_config()
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY
    os.environ["NUMEXPR_MAX_THREADS"] = NUMEXPR_MAX_THREADS
    #os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
        
    print("Generando los chunks...")
    text_chunks = get_text_chunks_boe()
    print("Generando los chunks OK. Tamaño=" + str(len(text_chunks)))
    
    print("Generando los FAISS. Modelo=" + MODEL_EMBEDDINGS_GOOGLE)
    get_vector_store(text_chunks)
    print("Generando los FAISS OK.")
    
    print("Terminado.")
    
    
if __name__ == "__main__":
   main()