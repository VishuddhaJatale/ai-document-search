const API_BASE_URL = "http://localhost:8000";

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(
    `${API_BASE_URL}/documents/upload`,
    {
      method: "POST",
      body: formData,
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));

    throw new Error(
      errorData.detail || "Failed to upload document"
    );
  }

  return response.json();
}

export async function deleteDocument(documentId) {
  const response = await fetch(
    `${API_BASE_URL}/documents/${documentId}`,
    {
      method: "DELETE",
    }
  );

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));

    throw new Error(
      errorData.detail || "Failed to delete document"
    );
  }

  return response.json();
}

export async function getDocuments() {
  const response = await fetch(
    `${API_BASE_URL}/documents`
  );

  if (!response.ok) {
    throw new Error("Failed to load documents");
  }

  return response.json();
}

export async function queryDocuments(
  question,
  documentId = null,
  topK = 6
) {
  try {
    const response = await fetch(
      `${API_BASE_URL}/query`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          document_id: documentId,
          top_k: topK,
        }),
      }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));

      throw new Error(
        errorData.detail || "Failed to get answer"
      );
    }

    return response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error(
        "Unable to connect to the backend. Please make sure the server is running."
      );
    }

    throw error;
  }
}