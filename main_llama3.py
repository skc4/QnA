import argparse
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from app import RAGApp
import tkinter as tk


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_embedding_function():
    embeddings = OllamaEmbeddings(model="llama3")
    return embeddings


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\n\nSources: {sources}"
    print(formatted_response)
    return response_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="Question here")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


if __name__ == "__main__":
    # main()
    root = tk.Tk()
    app = RAGApp(root, query_rag, __file__, './data')
    root.mainloop()