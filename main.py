from utils import read_docs_from_file, chunk_splitter, embedding, response

input = """Сколько мне будут стоить смски с оповещениями об операциях"""

documents = read_docs_from_file()
chunks = chunk_splitter(document=documents)
context = embedding(chunks=chunks, input = input)
response(context=context, input=input)
