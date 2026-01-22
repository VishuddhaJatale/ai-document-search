from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

def load_documents(folder_path="data/uploaded_docs"):

    loader = DirectoryLoader(
    folder_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

    return loader.load()
