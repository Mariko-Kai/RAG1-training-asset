from utils import read_docs_from_file, chunk_splitter, embedding, response

input = """Когда осуществляется начисление процентов на остаток средств"""
k = 4

documents = read_docs_from_file()
chunks = chunk_splitter(document=documents)
context = embedding(chunks=chunks, input = input)
response(context=context, input=input)
