import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { queryDocuments } from "../services/api";
import SourceCard from "./SourceCard";

function Chat({ selectedDocument }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const clearChat = () => { if (!loading) setMessages([]); };

  const handleAsk = async (event) => {
    event.preventDefault();
    if (!question.trim() || loading) return;
    const userQuestion = question.trim();
    const history = messages.filter((message) => ["user", "assistant"].includes(message.role)).slice(-6);
    setMessages((current) => [...current, { role: "user", content: userQuestion }, { role: "assistant", content: "", sources: [], streaming: true }]);
    setQuestion("");
    setLoading(true);
    try {
      const result = await queryDocuments(userQuestion, selectedDocument?.document_id || null, 6, history, (partial) => {
        setMessages((current) => current.map((message, index) => index === current.length - 1 ? { ...message, content: partial } : message));
      });
      setMessages((current) => current.map((message, index) => index === current.length - 1 ? { role: "assistant", content: result.answer, sources: result.sources || [] } : message));
    } catch (error) {
      setMessages((current) => current.map((message, index) => index === current.length - 1 ? { role: "assistant", content: `Error: ${error.message}`, sources: [] } : message));
    } finally { setLoading(false); }
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-1 pr-2">
        {messages.length === 0 ? (
          <div className="flex h-full items-center justify-center"><div className="max-w-md text-center">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-700 bg-slate-900 text-3xl shadow-lg">✨</div>
            <h3 className="mt-6 text-xl font-semibold text-white">Ask your documents anything</h3>
            <p className="mt-3 text-sm leading-6 text-slate-400">Upload a PDF or text document and ask questions. The AI retrieves relevant sections and answers from your documents.</p>
            <div className="mt-6 flex flex-wrap justify-center gap-2"><span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-400">Semantic Search</span><span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-400">Source Citations</span><span className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs text-slate-400">AI Answers</span></div>
          </div></div>
        ) : <div className="space-y-6 py-4">
          <div className="flex justify-end"><button type="button" onClick={clearChat} disabled={loading} className="text-xs text-slate-500 hover:text-slate-200 disabled:opacity-40">Clear chat</button></div>
          {messages.map((message, index) => <div key={index} className={message.role === "user" ? "flex justify-end" : "flex justify-start"}>
            <div className={message.role === "user" ? "max-w-[80%] rounded-2xl rounded-br-md bg-blue-600 px-4 py-3 text-sm leading-6 text-white shadow-lg shadow-blue-950/20" : "max-w-[90%] rounded-2xl rounded-bl-md border border-slate-700 bg-[#172238] px-5 py-4 text-sm leading-6 text-slate-200"}>
              {message.role === "user" ? <p className="whitespace-pre-wrap">{message.content}</p> : <div className="markdown-content max-w-none"><ReactMarkdown>{message.content || (message.streaming ? "Searching your documents…" : "")}</ReactMarkdown></div>}
              {message.role === "assistant" && <SourceCard sources={message.sources} />}
            </div>
          </div>)}
          {loading && messages[messages.length - 1]?.content === "" && <p className="text-xs text-slate-500">Searching your documents...</p>}
        </div>}
      </div>
      <div className="shrink-0 border-t border-slate-800/80 bg-[#0F172A] pt-4">
        <form onSubmit={handleAsk} className="flex gap-3 rounded-xl border border-slate-700 bg-slate-950 p-2 transition focus-within:border-blue-500/60">
          <input type="text" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={selectedDocument ? `Ask about ${selectedDocument.filename}...` : "Ask about your documents..."} disabled={loading} className="min-w-0 flex-1 bg-transparent px-3 py-2 text-sm text-white outline-none placeholder:text-slate-600 disabled:cursor-not-allowed" />
          <button type="submit" disabled={!question.trim() || loading} className="shrink-0 rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40">{loading ? "..." : "Ask"}</button>
        </form>
        <p className="mt-2 text-center text-[11px] text-slate-600">Answers are generated from your uploaded documents</p>
      </div>
    </div>
  );
}
export default Chat;
