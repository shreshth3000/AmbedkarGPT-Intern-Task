from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
import warnings
#Deprecationwarnings were excessive
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Preprocess the input text and build the vector database.
def build_vectorstore(speech_path, persist_dir="chroma_db"):
    print("Loading speech and creating chunks")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    # Converts text chunks into numerical embeddings.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # stores embeddings inside Chromadb
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    vectordb.persist()
    return vectordb

# Create the retrieval + generation pipeline using the stored vectors and Mistral 7B.
def build_qa_chain(vectordb):
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    llm = Ollama(model="mistral")
    
    # Builds the RetrievalQA chain that performs RAG.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

#terminal interface
def cli_loop(qa_chain):
    print("Type 'exit' if you want to quit.")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit"):
            return
        result = qa_chain.invoke(query)
        
        print("Answer ", result["result"])

def main():
    speech_path = "speech.txt"
    vectordb = build_vectorstore(str(speech_path))
    qa_chain = build_qa_chain(vectordb)
    print("Starting CLI GPT")
    cli_loop(qa_chain)

main()