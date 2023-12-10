import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader


model_name = "BAAI/bge-large-en"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceBgeEmbeddings(model_name = model_name,
                                      model_kwargs = model_kwargs,
                                      encode_kwargs = encode_kwargs)



loader = DirectoryLoader('Investment-Banker-Chatbot-main\Investment-Banker-Chatbot-main\Data',glob = '**/*.pdf',show_progress = True,loader_cls=PyPDFLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 100)
texts = text_splitter.split_documents(documents)
#print(texts[0])

vector_store = Chroma.from_documents(texts,embeddings,collection_metadata = {'hnsw:space': 'cosine'},persist_directory = 'stores/banking_cosine')

print('vectore store created......')


model_name = "BAAI/bge-large-en"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings':False}
embeddings = HuggingFaceBgeEmbeddings(model_name = model_name,
                                      model_kwargs = model_kwargs,
                                      encode_kwargs = encode_kwargs)


















