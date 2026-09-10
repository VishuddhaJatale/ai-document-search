function DocumentList({
  documents,
  selectedDocument,
  onSelectDocument,
  onDeleteDocument,
  deletingDocumentId,
}) {
  return (
    <div className="flex-1 px-6 pb-6">
      {documents.length === 0 ? (
        <div className="flex h-full items-center justify-center">
          <div className="text-center">
            <div className="text-3xl opacity-40">
              📚
            </div>

            <p className="mt-3 text-sm text-slate-500">
              No documents uploaded yet
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          {documents.map((document) => {
            const isSelected =
              selectedDocument?.document_id === document.document_id;

            return (
              <div
                key={document.document_id}
                onClick={() => onSelectDocument(document)}
                className={`cursor-pointer rounded-xl border p-4 transition ${
                  isSelected
                    ? "border-blue-500/60 bg-blue-500/10 shadow-lg shadow-blue-950/20"
                    : "border-slate-800 bg-slate-950/50 hover:border-slate-700 hover:bg-slate-900/70"
                }`}
              >
                <div className="flex items-center justify-between gap-3">

                  <div className="flex min-w-0 items-center gap-3">

                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-500/10">
                      📄
                    </div>

                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium text-slate-200">
                        {document.filename}
                      </p>

                      <p className="mt-1 text-xs text-emerald-400">
                        ✓ Indexed successfully
                      </p>
                    </div>

                  </div>

                  <button
                    type="button"
                    disabled={deletingDocumentId === document.document_id}
                    onClick={(event) => {
                        event.stopPropagation();
                        onDeleteDocument(document.document_id);
                    }}
                    className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-red-500/30 bg-red-500/10 text-red-400 transition hover:border-red-500/60 hover:bg-red-500/20 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-50"
                    title={
                        deletingDocumentId === document.document_id
                        ? "Deleting..."
                        : "Delete document"
                    }
                    aria-label={`Delete ${document.filename}`}
                    >
                    {deletingDocumentId === document.document_id ? "⏳" : "🗑️"}
                    </button>

                </div>

                {isSelected && (
                  <div className="mt-3 flex items-center gap-2 text-xs text-blue-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                    Selected for questions
                  </div>
                )}

              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default DocumentList;