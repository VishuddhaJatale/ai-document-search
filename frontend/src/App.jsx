import { useEffect, useState } from "react";
import Header from "./components/Header";
import DocumentUpload from "./components/DocumentUpload";
import DocumentList from "./components/DocumentList";
import Chat from "./components/Chat";
import { deleteDocument, getDocuments, getHealth, setApiKey } from "./services/api";

function App() {
  const [documents, setDocuments] = useState([]);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [documentsError, setDocumentsError] = useState("");
  const [deletingDocumentId, setDeletingDocumentId] = useState(null);
  const [deleteError, setDeleteError] = useState("");
  const [keyRequired, setKeyRequired] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");
  const [authenticated, setAuthenticated] = useState(false);
  const [authError, setAuthError] = useState("");
  const [checkingBackend, setCheckingBackend] = useState(true);

  const loadDocuments = async () => {
    setDocumentsError("");
    try {
      const existingDocuments = await getDocuments();
      setDocuments(existingDocuments);
      setAuthenticated(true);
    } catch (error) {
      setDocumentsError(error.message || "Failed to load documents");
      if (error.message?.includes("API key")) setAuthenticated(false);
    }
  };

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const health = await getHealth();
        if (!active) return;
        setKeyRequired(Boolean(health.api_key_required));
        if (!health.api_key_required) {
          await loadDocuments();
        } else if (sessionStorage.getItem("adi_api_key")) {
          await loadDocuments();
        }
      } catch (error) {
        if (active) setDocumentsError(error.message || "Unable to connect to backend");
      } finally {
        if (active) setCheckingBackend(false);
      }
    })();
    return () => { active = false; };
  }, []);

  const handleKeySubmit = async (event) => {
    event.preventDefault();
    setAuthError("");
    setApiKey(apiKeyInput);
    try {
      const docs = await getDocuments();
      setDocuments(docs);
      setAuthenticated(true);
      setApiKeyInput("");
    } catch (error) {
      setApiKey("");
      setAuthenticated(false);
      setAuthError(error.message || "Invalid API key");
    }
  };

  const handleUploadSuccess = (document) => {
    setDocuments((current) => current.some((item) => item.document_id === document.document_id) ? current : [...current, document]);
  };

  const handleDeleteDocument = async (documentId) => {
    if (deletingDocumentId) return;
    setDeletingDocumentId(documentId);
    setDeleteError("");
    try {
      await deleteDocument(documentId);
      setDocuments((current) => current.filter((document) => document.document_id !== documentId));
      if (selectedDocument?.document_id === documentId) setSelectedDocument(null);
    } catch (error) {
      setDeleteError(error.message || "Unable to delete document.");
    } finally {
      setDeletingDocumentId(null);
    }
  };

  if (checkingBackend) {
    return <div className="min-h-screen bg-[#080D1A] p-10 text-center text-slate-300">Connecting to AI Document Intelligence…</div>;
  }

  if (keyRequired && !authenticated) {
    return (
      <div className="min-h-screen bg-[#080D1A] text-slate-100">
        <Header />
        <main className="mx-auto flex min-h-[75vh] max-w-md items-center px-6">
          <form onSubmit={handleKeySubmit} className="w-full rounded-2xl border border-slate-800 bg-[#0F172A] p-7 shadow-xl">
            <h1 className="text-xl font-semibold text-white">API key required</h1>
            <p className="mt-2 text-sm leading-6 text-slate-400">Enter the API key configured by the administrator. It is kept only for this browser session.</p>
            <label htmlFor="api-key" className="mt-6 block text-sm text-slate-300">API key</label>
            <input id="api-key" type="password" autoComplete="current-password" value={apiKeyInput} onChange={(event) => setApiKeyInput(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white outline-none focus:border-blue-500" required />
            {authError && <p role="alert" className="mt-3 text-sm text-red-400">{authError}</p>}
            <button type="submit" className="mt-5 w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-blue-500">Continue</button>
          </form>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#080D1A] text-slate-100">
      <Header />
      <main className="mx-auto max-w-7xl px-6 py-8">
        {documentsError && <div role="alert" className="mb-4 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-sm text-red-400">{documentsError}</div>}
        <div className="grid h-[calc(100vh-140px)] min-h-0 grid-cols-1 gap-6 lg:grid-cols-3">
          <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-[#0F172A] shadow-xl shadow-black/10">
            <div className="shrink-0 border-b border-slate-800/80 p-6">
              <div className="flex items-center justify-between">
                <div><h2 className="text-lg font-semibold text-white">Documents</h2><p className="mt-1 text-sm text-slate-400">Your knowledge base</p></div>
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/10 text-blue-400">📄</div>
              </div>
            </div>
            <div className="shrink-0 p-6">
              <DocumentUpload onUploadSuccess={handleUploadSuccess} />
              {documentsError && <div className="mt-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs leading-5 text-red-400">{documentsError}</div>}
            </div>
            {deleteError && <p role="alert" className="mx-6 mb-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs text-red-400">{deleteError}</p>}
            <DocumentList documents={documents} selectedDocument={selectedDocument} onSelectDocument={setSelectedDocument} onDeleteDocument={handleDeleteDocument} deletingDocumentId={deletingDocumentId} />
          </section>
          <section className="flex min-h-0 flex-col overflow-hidden rounded-2xl border border-slate-800/80 bg-[#0F172A] shadow-xl shadow-black/10 lg:col-span-2">
            <div className="flex items-center justify-between border-b border-slate-800/80 px-6 py-5">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 text-lg">✨</div>
                <div><h2 className="font-semibold text-white">Document Assistant</h2><p className="text-xs text-slate-500">{selectedDocument ? `Searching in ${selectedDocument.filename}` : "Ask questions about your documents"}</p></div>
              </div>
              <div className="flex items-center gap-2">
                {selectedDocument && <button type="button" onClick={() => setSelectedDocument(null)} className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 transition hover:border-slate-600 hover:bg-slate-800 hover:text-slate-200">Deselect</button>}
                {keyRequired && <button type="button" onClick={() => { setApiKey(""); setAuthenticated(false); setDocuments([]); }} className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 hover:text-white">Sign out</button>}
                <div className="hidden rounded-full border border-slate-700 bg-slate-900 px-3 py-1.5 text-xs text-slate-400 sm:block">RAG Search</div>
              </div>
            </div>
            <div className="min-h-0 flex-1 p-5"><Chat selectedDocument={selectedDocument} /></div>
          </section>
        </div>
      </main>
    </div>
  );
}
export default App;
