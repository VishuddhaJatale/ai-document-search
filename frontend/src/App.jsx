import { useEffect, useState } from "react";
import Header from "./components/Header";
import DocumentUpload from "./components/DocumentUpload";
import DocumentList from "./components/DocumentList";
import Chat from "./components/Chat";
import { deleteDocument, getDocuments } from "./services/api";


function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documentsError, setDocumentsError] = useState("");
  const [deletingDocumentId, setDeletingDocumentId] = useState(null);

  useEffect(() => {
    const loadDocuments = async () => {
  try {
    setDocumentsError("");

    const existingDocuments = await getDocuments();
        setDocuments(existingDocuments);
      } catch (error) {
        console.error("Failed to load documents:", error);
        setDocumentsError(
          error.message || "Failed to load documents"
        );
      }
    };

    loadDocuments();
  }, []);

const handleUploadSuccess = (document) => {
  setDocuments((currentDocuments) => {
    const alreadyExists = currentDocuments.some(
      (currentDocument) =>
        currentDocument.document_id === document.document_id
    );

    if (alreadyExists) {
      return currentDocuments;
    }

    return [...currentDocuments, document];
  });
};

  const handleDeleteDocument = async (documentId) => {
  if (deletingDocumentId) {
    return;
  }

  setDeletingDocumentId(documentId);

  try {
    await deleteDocument(documentId);

    setDocuments((currentDocuments) =>
        currentDocuments.filter(
          (document) => document.document_id !== documentId
        )
      );

      if (selectedDocument?.document_id === documentId) {
        setSelectedDocument(null);
      }
    } catch (error) {
      console.error("Delete error:", error);
      alert(error.message);
    } finally {
      setDeletingDocumentId(null);
    }
  };

  return (
    <div className="min-h-screen bg-[#080D1A] text-slate-100">
      <Header />

      <main className="mx-auto max-w-7xl px-6 py-8">
        <div className="grid h-[calc(100vh-140px)] min-h-0 grid-cols-1 gap-6 lg:grid-cols-3">

          {/* Documents Panel */}
          <section className="flex flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-[#0F172A] shadow-xl shadow-black/10">

            {/* Panel Header */}
            <div className="border-b border-slate-800/80 p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold text-white">
                    Documents
                  </h2>

                  <p className="mt-1 text-sm text-slate-400">
                    Your knowledge base
                  </p>
                </div>

                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">
                  📄
                </div>
              </div>
            </div>

            {/* Upload Area */}
            <div className="p-6">
              <DocumentUpload onUploadSuccess={handleUploadSuccess} />
              {documentsError && (
                <div className="mt-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs leading-5 text-red-400">
                  {documentsError}
                </div>
              )}
            </div>

            <DocumentList
              documents={documents}
              selectedDocument={selectedDocument}
              onSelectDocument={setSelectedDocument}
              onDeleteDocument={handleDeleteDocument}
              deletingDocumentId={deletingDocumentId}
            />
          </section>


          {/* Chat Panel */}
          <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-[#0F172A] shadow-xl shadow-black/10 lg:col-span-2">
            {/* Chat Header */}
            <div className="flex items-center justify-between border-b border-slate-800/80 px-6 py-5">

              <div className="flex items-center gap-3">

                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 text-lg">
                  ✨
                </div>

                <div>
                  <h2 className="font-semibold text-white">
                    Document Assistant
                  </h2>

                  <p className="text-xs text-slate-500">
                    {selectedDocument
                      ? `Searching in ${selectedDocument.filename}`
                      : "Ask questions about your documents"}
                  </p>
                </div>

              </div>

              <div className="flex items-center gap-2">
                {selectedDocument && (
                  <button
                    type="button"
                    onClick={() => setSelectedDocument(null)}
                    className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 transition hover:border-slate-600 hover:bg-slate-800 hover:text-slate-200"
                  >
                    Deselect
                  </button>
                )}

                <div className="hidden rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 sm:block">
                  RAG Search
                </div>
              </div>
            </div>

            <div className="min-h-0 flex-1 p-5">
              <Chat selectedDocument={selectedDocument} />
            </div>

          </section>

        </div>
      </main>
    </div>
  );
}

export default App;