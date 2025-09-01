import { useEffect, useRef, useState } from "react";
import { askBackend } from "./lib/api";
import "./styles.css";

function Bubble({ role, text }) {
  return (
    <div className={`msg ${role === "user" ? "user" : "bot"}`}>
      <div className="bubble-head">
        <span>{role === "user" ? "🧑‍💻 You" : "🤖 Asistant"}</span>
      </div>
      <div>{text}</div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello! I can answer questions based on your PDFs using RAG. What would you like to ask?" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function onSend() {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setLoading(true);
    try {
      const answer = await askBackend(q);
      setMessages((m) => [...m, { role: "bot", text: answer }]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "bot", text: `Üzgünüm, bir hata oluştu: ${e.message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onSend();
  }

  function copyLast() {
    const last = [...messages].reverse().find((m) => m.role === "bot");
    if (last) {
      navigator.clipboard.writeText(last.text);
    }
  }

  function retry() {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (lastUser) {
      setInput(lastUser.text);
    }
  }

  return (
    <div className="container">
      <div className="chat">
        <div className="header">
          <div className="dot" />
          <div className="title">RAG PDF Chatbot</div>
          <div className="sub">Qdrant + FastAPI</div>
          <div className="tools">
            <button className="tool" onClick={copyLast} title="Son yanıtı kopyala">Copy</button>
            <button className="tool" onClick={retry} title="Son soruyu düzenle">Rewrite</button>
          </div>
        </div>

        <div className="messages" ref={listRef}>
          {messages.map((m, i) => <Bubble key={i} role={m.role} text={m.text} />)}
          {loading && (
            <div className="msg bot">
              <div className="bubble-head"><span>🤖 Asistan</span></div>
              <div className="typing">
                <span className="dotty"></span><span className="dotty"></span><span className="dotty"></span>
              </div>
            </div>
          )}
        </div>

        <div className="inputbar">
          <textarea
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask query… (Ctrl/⌘ + Enter)"
          />
          <button onClick={onSend} disabled={loading}>Send</button>
        </div>

        <div className="footer">
          <span className="kbd">Ctrl/⌘ + Enter</span> ile gönder
        </div>
      </div>
    </div>
  );
}
