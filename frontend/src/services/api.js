const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function parseError(response, fallback) {
  const errorData = await response.json().catch(() => ({}));
  throw new Error(errorData.detail || fallback);
}

export async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/documents/upload`, { method: "POST", body: formData });
  if (!response.ok) await parseError(response, "Failed to upload document");
  return response.json();
}

export async function deleteDocument(documentId) {
  const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, { method: "DELETE" });
  if (!response.ok) await parseError(response, "Failed to delete document");
  return response.json();
}

export async function getDocuments() {
  const response = await fetch(`${API_BASE_URL}/documents`);
  if (!response.ok) throw new Error("Failed to load documents");
  return response.json();
}

export async function queryDocuments(question, documentId = null, topK = 6, history = [], onToken = null) {
  const payload = { question, document_id: documentId, top_k: topK, history: history.slice(-6) };
  try {
    if (onToken) {
      try {
        const streamResponse = await fetch(`${API_BASE_URL}/query/stream`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
        });
        if (!streamResponse.ok || !streamResponse.body) throw new Error("Streaming unavailable");
        const reader = streamResponse.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let answer = "";
        let sources = [];
        let finalReceived = false;
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";
          for (const eventText of events) {
            const eventName = eventText.match(/^event: (.+)$/m)?.[1];
            const dataLine = eventText.match(/^data: (.+)$/m)?.[1];
            if (!dataLine) continue;
            const data = JSON.parse(dataLine);
            if (eventName === "token") { answer += data; onToken(answer); }
            if (eventName === "final") { sources = data.sources || []; finalReceived = true; }
            if (eventName === "error") throw new Error(data.detail || "Streaming failed");
          }
        }
        if (!answer || !finalReceived) throw new Error("Incomplete streaming response");
        return { answer, sources };
      } catch {
        // A failed or interrupted stream falls back to the stable JSON endpoint.
      }
    }
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!response.ok) await parseError(response, "Failed to get answer");
    return response.json();
  } catch (error) {
    if (error instanceof TypeError) throw new Error("Unable to connect to the backend. Please make sure the server is running.");
    throw error;
  }
}
