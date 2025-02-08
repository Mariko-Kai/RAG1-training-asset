from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import TFIDFRetriever
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import os


def clean_text(text):
    return text.replace("\n", "").replace("\xa0", " ")

def get_id_key(doc):
    return doc.metadata.get("id", 0)

def read_docs_from_file():
    doc1 = PyPDFLoader("documents/doc1.pdf")
    doc2 = PyPDFLoader("documents/doc2.pdf")
    doc3 = CSVLoader(file_path="documents/cards.csv", encoding='utf-8')
    document = doc1.load()
    document.extend(doc2.load())
    document.extend(doc3.load())
    for doc in document:
        doc.page_content = clean_text(doc.page_content)

    return document

def chunk_splitter(document: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,

    )
    chunks = text_splitter.split_documents(document)

    last_page_id = None
    chunk_id = 0
    for chunk in chunks:
        chunk.metadata["id"] = str(chunk_id)     
        chunk_id += 1

    return chunks


def embedding(chunks: list[Document], input: str):
    k = 4
    TFIDF_local_path = "TFIDF_vectors"
    FAISS_local_path = "faiss_vectorsv4"
    embedding_model = OllamaEmbeddings(model="snowflake-arctic-embed2:568m")

    chunk_id_list = [chunk.metadata["id"] for chunk in chunks]
    if os.path.isdir(TFIDF_local_path) and os.listdir(TFIDF_local_path):
        TFIDF_vector_db = TFIDFRetriever.load_local(TFIDF_local_path, allow_dangerous_deserialization=True)
    else:
        TFIDF_vector_db = TFIDFRetriever.from_documents(chunks)
        TFIDF_vector_db.save_local(TFIDF_local_path)

    if os.path.isdir(FAISS_local_path) and os.listdir(FAISS_local_path):
        FAISS_vector_db = FAISS.load_local(FAISS_local_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        FAISS_vector_db = FAISS.from_documents(chunks, embedding_model, ids = chunk_id_list, distance_strategy="EUCLIDEAN")
        FAISS_vector_db.save_local(FAISS_local_path)

    results = FAISS_vector_db.similarity_search(input, k = k)

    TF_results = TFIDF_vector_db.invoke(input)

    results.extend(TF_results)
    index = list(dict.fromkeys([doc.metadata.get("id") for doc in results ]))
    results = [doc for doc in results if doc.metadata.get("id") in index]

    result = None
    list_index = list()
    for i in range(len(results)):
        list_index.append(str(int(results[i].metadata["id"]) + 1))
    for i in results:
        result = [doc for doc in chunks if doc.metadata["id"] in list_index]
    results.extend(result)
    results_sorted = sorted(results, key=get_id_key)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results_sorted])

    return context_text



def response(context, input):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """Ответьте на вопрос, исходя из контекста ниже. Ответ должен быть кратким и лаконичным.
        Ответьте «Я не знаю.», если не уверены в ответе. {context}.
        """),("human","""
        Вопрос: {question}""")
    ])
    prompt = prompt_template.format(context=context, question=input)

    model = OllamaLLM(model="llama3.1:8b", stream = True)
    response_data = [input, context]
    #print(f"""{response_data}\n
    #      ------------------------------------""")

    response = model.stream(prompt)
    for token in response:
        print(token, end="", flush=True) 