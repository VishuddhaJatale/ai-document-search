from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

def load_documents(folder_path="data/uploaded_docs"):

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    return loader.load()
