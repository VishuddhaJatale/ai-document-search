import { useRef, useState } from "react";
import { uploadDocument } from "../services/api";

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

function DocumentUpload({ onUploadSuccess }) {
  const fileInputRef = useRef(null);

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = async (event) => {
    const file = event.target.files[0];

    if (!file) {
      return;
    }

    setError("");

    // Validate file type
    const fileName = file.name.toLowerCase();

    if (!fileName.endsWith(".pdf") && !fileName.endsWith(".txt")) {
      setError("Only PDF and TXT files are supported.");
      event.target.value = "";
      return;
    }

    // Validate file size
    if (file.size > MAX_FILE_SIZE) {
      setError("File size must be less than 10 MB.");
      event.target.value = "";
      return;
    }

    // Validate empty file
    if (file.size === 0) {
      setError("The selected file is empty.");
      event.target.value = "";
      return;
    }

    setUploading(true);

    try {
      const result = await uploadDocument(file);

      onUploadSuccess(result);
    } catch (error) {
      setError(error.message);
    } finally {
      setUploading(false);

      // Allows selecting the same file again
      event.target.value = "";
    }
  };

  const handleChooseFile = () => {
    if (!uploading) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf,.txt"
        onChange={handleFileChange}
        className="hidden"
      />

      <div
        onClick={handleChooseFile}
        className={`group rounded-xl border border-dashed border-slate-700 bg-slate-950/40 p-8 text-center transition-all duration-200 ${
          uploading
            ? "cursor-not-allowed opacity-70"
            : "cursor-pointer hover:border-blue-500/60 hover:bg-blue-500/5"
        }`}
      >
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-blue-500/10 text-2xl">
          {uploading ? "⏳" : "⬆️"}
        </div>

        <h3 className="mt-4 text-sm font-medium text-slate-200">
          {uploading ? "Uploading..." : "Upload a document"}
        </h3>

        <p className="mt-2 text-xs leading-5 text-slate-500">
          {uploading
            ? "Processing and indexing your document"
            : "Drag & drop or click to browse"}

          {!uploading && (
            <>
              <br />
              PDF or TXT files · Max 10 MB
            </>
          )}
        </p>

        {!uploading && (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              handleChooseFile();
            }}
            className="mt-5 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-lg shadow-blue-600/20 transition hover:bg-blue-500"
          >
            Choose File
          </button>
        )}
      </div>

      {error && (
        <p className="mt-3 rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-xs leading-5 text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

export default DocumentUpload;