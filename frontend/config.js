// Where the backend lives.
// Empty = the same server that served this page (local: uvicorn serves the dashboard).
// On Vercel the page is static, so it must point at the deployed backend.
window.MC_API_BASE = location.hostname.endsWith(".vercel.app")
  ? "https://mission-control-ai-v3b6.onrender.com"
  : "";
