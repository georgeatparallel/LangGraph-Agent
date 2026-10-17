# 🤖 LangChain + LangGraph Research Agent

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-FF6B6B?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036?style=for-the-badge&logo=groq&logoColor=white)
![DuckDuckGo](https://img.shields.io/badge/DuckDuckGo-Web_Search-DE5833?style=for-the-badge&logo=duckduckgo&logoColor=white)
![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Parsing-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> A practical AI assistant that searches the web and reads PDFs — right from your terminal.

This project combines **LangChain**, **LangGraph**, and **Groq** to build a tool-using conversational agent that maintains session history, intelligently routes tool calls, and delivers clear, long-form responses. A ready-to-use chess reference (`sample.pdf`) is included so you can test document Q&A out of the box.

---

## ✨ Features

- 🔍 **Web search** via DuckDuckGo for fresh, real-time information
- 📄 **PDF reading** with page-by-page text extraction for document Q&A
- 🔄 **LangGraph workflow** with automatic tool routing and loop-back execution
- 🧠 **Persistent conversation memory** saved to disk and restored on restart
- 📡 **Live decision tracing** for LLM calls, tool calls, and errors in terminal logs
- 💰 **Token and cost tracking** with per-session usage totals and estimated API spend
- 📏 **Minimum response length** enforced at 100+ words via system prompt rules

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Agent Framework | LangChain + LangGraph |
| LLM Provider | Groq API (`llama-3.1-8b-instant`) |
| Web Search | DuckDuckGo (`ddgs`), or opt-in Parallel Search MCP |
| PDF Parsing | PyMuPDF (`pymupdf`) |
| Config | Python Dotenv |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Langchain_Agent
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**
```bat
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the agent

```bash
python main.py
```

You should see:
```
Agent ready! Type clear to reset memory or quit to exit.
```

Command reference:
- `quit` / `exit` / `q`: saves conversation history, prints token summary, and exits
- `clear`: clears persisted chat history and resets current in-memory history

On exit, a session summary is printed with total LLM calls, input/output tokens, and estimated USD cost.

Conversation memory is persisted in `chat_history.json` at the project root.

### Optional: Parallel web search

To use [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
for current information with source URLs and excerpts, install the optional dependencies:

```bash
pip install -r requirements-parallel.txt
```

Add this setting to your existing `.env` file, alongside `GROQ_API_KEY`:

```env
WEB_SEARCH_PROVIDER=parallel
```

Then run `python main.py` as usual and try: `Search the web for recent LangGraph releases and include source URLs.`
The existing `web_search` tool uses Parallel's Streamable HTTP MCP endpoint at
`https://search.parallel.ai/mcp`. Anonymous search is free for light use, uses Fast
mode, and needs no Parallel account or API key. The Groq key is still required to
run the conversational agent. This option adds web search only.

Leave `WEB_SEARCH_PROVIDER` unset, or set it to `duckduckgo`, to use the existing
DuckDuckGo search. PDF reading and conversation history work the same way with
either provider. Only `duckduckgo` and `parallel` are accepted provider values.

To check the search tool without calling the LLM:

```bash
python -c "from parallel_search import parallel_web_search; print(parallel_web_search.invoke({'query': 'LangGraph latest release'}))"
```

Search failures, including service limits, are returned as `Search failed: ...`
tool results. Parallel requests have a 30-second timeout. If the service reports
a rate limit, wait as directed before retrying. If Python cannot import `mcp`,
install `requirements-parallel.txt` in the same environment used to run the agent.

---

## 📄 Demo: Chess Book PDF

The repo includes `sample.pdf` (a chess book) for immediate document-aware testing.

**Try these prompts:**

- `"Summarize the key ideas in sample.pdf."`
- `"What opening principles are covered in the chess book?"`
- `"Give me five practical tactics from sample.pdf with simple examples."`
- `"Create a beginner-friendly chess study plan based on sample.pdf."`

**How PDF extraction works:**
- Extracts text page by page
- Inserts `--- Page N ---` separators between pages
- Truncates very large outputs for stability
- Returns clear error messages for missing or non-extractable files

---

## 🏗️ How It Works

The agent is built as a compact **LangGraph** state machine with two nodes:

| Node | Responsibility |
|---|---|
| `agent` | Builds messages from system prompt + history, invokes LLM with tools |
| `tools` | Executes selected tool calls and returns results to the agent |

**Execution flow:**

```
Start → agent → (tool needed?) → tools → agent → ... → End
```

This keeps tool use controlled and allows the model to ground final answers in retrieved content.

---

## 📁 Project Layout

```
Langchain_Agent/
├── main.py          # CLI loop, persistence flow, and token summary output
├── agent.py         # LLM setup, system prompt, graph wiring
├── logger.py        # Live callback logs for LLM/tool activity
├── token_tracker.py # Token accounting and cost estimation
├── memory.py        # Save/load/clear persistent chat history
├── tools.py         # web_search and read_pdf tool implementations
├── requirements.txt # Python dependencies
├── chat_history.json# Auto-created persistent conversation store
└── sample.pdf       # Chess book for PDF Q&A demos
```

---

## 📊 Observability and Cost Tracking

This project now includes runtime transparency and usage accounting:

- **Live trace logs** come from `AgentLogger` and include:
	- LLM call start/end
	- Tool start/input/end
	- Chain-level errors
- **Token tracking** records `input_tokens` and `output_tokens` from `usage_metadata`.
- **Cost estimation** uses model pricing defined in `token_tracker.py`.

> Pricing values are examples and can change over time. Verify current rates in the Groq pricing console.

---

## 🛠️ Troubleshooting

**Groq key / auth issues**
- Confirm `.env` exists in the project root
- Verify `GROQ_API_KEY` is valid at [console.groq.com](https://console.groq.com)
- Restart your terminal after editing `.env`

**Web search returns no results**
- Check your internet connection
- Try a more specific or rephrased query
- Verify DuckDuckGo is accessible from your network

**Memory did not resume as expected**
- Check whether `chat_history.json` exists in the project root
- Confirm you exited with `quit`, `exit`, or `q` so history was saved
- If needed, delete `chat_history.json` and start a fresh session

**PDF errors or weak text extraction**
- Verify the file path and filename are correct
- Keep `sample.pdf` in the project root, or provide an absolute path
- For image-only / scanned PDFs, run OCR preprocessing first

---

## 🔐 Security Notes

- **Never commit real API keys** to version control
- Store all secrets in `.env` and ensure it is listed in `.gitignore`
- Treat tool output as helpful context — not guaranteed truth

---

## 🗺️ Roadmap

- [ ] Streaming token output in the CLI
- [ ] Page-range support for selective PDF reading
- [ ] Source citations with URL and page references
- [ ] Unit tests for tools and graph behavior
- [ ] Optional web UI (Streamlit or FastAPI frontend)

---

## 📜 License

Add your preferred license (e.g. MIT) before publishing.
