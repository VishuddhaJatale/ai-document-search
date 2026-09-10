function SourceCard({ sources }) {
  if (!sources || sources.length === 0) {
    return null;
  }

  const groupedSources = {};

  sources.forEach((source) => {
    const filename = source.filename || "Unknown document";

    if (!groupedSources[filename]) {
      groupedSources[filename] = {
        filename,
        pages: new Set(),
      };
    }

    if (source.page !== undefined && source.page !== null) {
      groupedSources[filename].pages.add(source.page);
    }
  });

  const documents = Object.values(groupedSources).map((document) => ({
    ...document,
    pages: Array.from(document.pages).sort((a, b) => a - b),
  }));

  const uniquePages = new Set(
    sources
      .map((source) => source.page)
      .filter(
        (page) => page !== undefined && page !== null
      )
  );

  return (
    <div className="mt-5 border-t border-slate-700 pt-4">

      {/* Sources Header */}
      <div className="mb-3 flex items-center justify-between">

        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Sources
          </span>

          <span className="rounded-full bg-slate-700/60 px-2 py-0.5 text-[10px] text-slate-400">
            {uniquePages.size}{" "}
            {uniquePages.size === 1 ? "page" : "pages"}
          </span>
        </div>

        <span className="text-[10px] text-slate-600">
          {sources.length} relevant{" "}
          {sources.length === 1 ? "section" : "sections"}
        </span>

      </div>

      {/* Source Documents */}
      <div className="space-y-2">

        {documents.map((document, index) => (
          <div
            key={index}
            className="rounded-xl border border-slate-700 bg-slate-900/70 px-4 py-3 transition hover:border-slate-600"
          >

            <div className="flex items-start gap-3">

              {/* File Icon */}
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-blue-500/10 text-sm">
                📄
              </div>

              {/* File Information */}
              <div className="min-w-0 flex-1">

                <p className="break-words text-xs font-medium leading-5 text-slate-200">
                  {document.filename}
                </p>

                {document.pages.length > 0 && (
                  <p className="mt-1.5 text-xs text-slate-500">
                    {document.pages.length === 1
                      ? "Page"
                      : "Pages"}{" "}
                    {document.pages.join(" · ")}
                  </p>
                )}

              </div>

            </div>

          </div>
        ))}

      </div>

    </div>
  );
}

export default SourceCard;