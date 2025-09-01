export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export async function askBackend(query) {
  const resp = await fetch(`${API_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  const data = await resp.json();
  return data.answer;
}
