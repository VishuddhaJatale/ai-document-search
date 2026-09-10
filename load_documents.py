import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredPDFLoader
)

def load_documents(folder_path):

    documents = []

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        try:

            if file.lower().endswith(".pdf"):

                loader = PyPDFLoader(file_path)
                docs = loader.load()

                if not docs:
                    loader = UnstructuredPDFLoader(file_path, strategy="hi_res")
                    docs = loader.load()

            elif file.lower().endswith(".txt"):

                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()

            else:
                continue

            for doc in docs:
                doc.metadata["source"] = file

            documents.extend(docs)

        except Exception as e:
            print("Skipped file:", file, e)

    return documents
