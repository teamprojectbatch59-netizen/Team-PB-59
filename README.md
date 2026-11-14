# Ruas AI

A local-first chatbot: plain HTML/CSS/JS frontend, Python Flask backend, and
[Ollama](https://ollama.com) models running on your own machine. Chat and file
data stay in the local Ruas AI installation. New accounts use a configured
SMTP relay to send an email verification code.

## Features

- Streaming chat with Markdown, code blocks and copy buttons
- Four color themes (Midnight, Light, Forest and Sunset), with saved creativity and response-speed controls
- Fast, Balanced and Deep response profiles, with context/output limits and warm model reuse for quicker follow-up replies
- Refresh the installed Ollama model list from the sidebar after adding a model
- Searchable, pinnable chat history grouped by date, with a compact desktop rail and keyboard-friendly mobile drawer
- Turn numeric Markdown tables into animated bar, line or donut charts; switch between
  original values, running totals and percentages, then save the chart as SVG
- Two-model setup:
  - **`llama3.2:3b`** (~2.0 GB) for fast everyday text chat
  - **`qwen3-vl:8b`** (~6.2 GB) for images and document reading
- Attach images (PNG/JPG/WebP/GIF/BMP/TIFF) and documents (PDF, DOCX, XLSX,
  CSV, TXT, Markdown, code, and more)
- Drag & drop, paste, and multi-file uploads
- Image preview cards and document upload rows with live progress and cancel controls
- Automatic model routing: image turns go to the vision model; text stays on
  the fast model
- Chat history saved in the browser (localStorage), including the last open chat, theme and model preferences, mobile
  friendly
- Uploads are stored locally and cleaned up automatically after 12 hours
- Email-verified accounts with per-account chats, settings and file access

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) running locally (`ollama serve`)
- About 9 GB of disk space for the two models
- A GPU is optional. On a 6 GB laptop GPU the 8B vision model partially
  offloads to the CPU, so image answers are slower than text chat.

## Quick start (Windows)

1. Install Ollama and make sure it is running.
2. To enable signup, create a local environment file and edit it in Notepad:

   ```powershell
   Copy-Item env.example .env
   notepad .env
   ```

   Enter your SMTP credentials and a stable `SECRET_KEY` in `.env`. For Gmail,
   use an App Password with 2-Step Verification enabled. Keep `.env` private;
   it is excluded from source control. Port 587 uses STARTTLS by default and
   port 465 uses implicit TLS. Restart Ruas AI after editing `.env`. Without
   SMTP settings, signup displays a setup error and existing accounts can
   still sign in.

3. In the same PowerShell window, run `.\run.bat`. It finds Python, creates
   `.venv`, installs dependencies, and starts the server.
4. Pull the models (first run only). Either let `run.bat` prompt you, or run:

   ```powershell
   .\pull_models.ps1
   ```

   or manually:

   ```powershell
   ollama pull llama3.2:3b
   ollama pull qwen3-vl:8b
   ollama pull llama-guard3:1b
   ```

5. Open <http://127.0.0.1:5000>.

## Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

## Models

| Purpose | Model | Size | Notes |
| --- | --- | --- | --- |
| Text chat | `llama3.2:3b` | ~2.0 GB | Fast, runs well on modest hardware |
| Images & files | `qwen3-vl:8b` | ~6.2 GB | Vision + document understanding |

> **Why not "Llama 3 2B"?** Llama 3.2 ships in 1B and 3B sizes; there is no 2B
> build. `llama3.2:3b` is the closest match and is the model this app uses.

> **Why Qwen 8B?** `qwen3-vl:8b` reads images *and* long documents, so a single
> model covers both file and image questions. Qwen's "thinking" mode is disabled
> by the backend for snappy replies.

### Routing rules

- Plain chat → `llama3.2:3b`
- Any message with an image → `qwen3-vl:8b`
- Document-only turns in Fast/Balanced → the small text model; Deep may use Qwen for longer document context
- Choosing a specific model in the dropdown overrides the text model, but
  images are always sent to a vision-capable model
- If no vision model is installed, image uploads return a clear error telling
  you which model to pull

## Supported files

| Type | Extensions |
| --- | --- |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.gif` `.bmp` `.tif` `.tiff` |
| Documents | `.pdf` `.docx` `.xlsx` |
| Text / data | `.txt` `.md` `.csv` `.tsv` `.json` `.jsonl` `.xml` `.yaml` `.yml` `.log` `.sql` |
| Code | `.py` `.js` `.ts` `.html` `.css` `.go` `.rs` `.java` `.c` `.cpp` `.cs` `.php` `.sh` `.ps1` and more |

Limits: 20 MB per file, 4 images and 8 attachments per request, 30,000
characters per document, ~24,000 characters of context per request.

## Configuration

Set environment variables before starting the server:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API address |
| `TEXT_MODEL` | `llama3.2:3b` | Default text model |
| `VISION_MODEL` | `qwen3-vl:8b` | Model for images and documents |
| `OLLAMA_NUM_CTX` | `8192` | Context window size |
| `OLLAMA_KEEP_ALIVE` | `10m` | How long Ollama keeps a model loaded between chats |
| `MAX_UPLOAD_MB` | `20` | Per-file upload limit |
| `PORT` | `5000` | Web server port |
| `SECRET_KEY` | Random per launch | Stable Flask session and verification-code signing key |
| `SESSION_COOKIE_SECURE` | `0` | Set to `1` when serving over HTTPS |
| `SMTP_HOST` | — | SMTP server hostname |
| `SMTP_PORT` | `587` | SMTP server port (`465` for implicit TLS) |
| `SMTP_USERNAME` | — | SMTP AUTH username |
| `SMTP_PASSWORD` | — | SMTP AUTH password or provider credential |
| `SMTP_FROM_EMAIL` | `SMTP_USERNAME` | Sender address for verification mail |
| `SMTP_FROM_NAME` | `Ruas AI` | Sender display name |
| `SMTP_USE_TLS` | `1` on port 587 | Use STARTTLS; required unless SSL is enabled |
| `SMTP_USE_SSL` | `1` on port 465 | Use implicit TLS |
| `SMTP_TIMEOUT` | `10` | SMTP connection timeout in seconds |

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Chat UI |
| `GET` | `/login` | Sign-in page |
| `GET` | `/signup` | Account creation and email verification page |
| `POST` | `/api/auth/signup` | Create pending account and send verification code |
| `POST` | `/api/auth/verify` | Verify email and sign in |
| `POST` | `/api/auth/login` | Sign in with verified account credentials |
| `POST` | `/api/auth/logout` | End the current session |
| `GET` | `/api/health` | Ollama reachability check |
| `GET` | `/api/models` | Installed models, chosen text/vision models, readiness |
| `POST` | `/api/upload` | Upload files (`multipart/form-data`, field `files`) |
| `POST` | `/api/chat` | Stream a chat reply as Server-Sent Events |

Workspace APIs require a signed-in session and same-session CSRF token. Passwords
are stored as password hashes. Accounts and per-user settings are stored in
`ruas_auth.sqlite3`; account uploads are tagged to their owner. Chats and profile
preferences are stored in browser storage scoped to the signed-in account.

`POST /api/chat` body:

```json
{
  "model": "auto",
  "performance_profile": "fast",
  "messages": [
    {
      "role": "user",
      "content": "Summarise this file",
      "attachments": [{ "id": "<upload id>" }]
    }
  ],
  "temperature": 0.7
}
```

`temperature` is optional and defaults to `0.7`; the UI exposes it as a saved
creativity slider from `0.0` (more focused) to `1.5` (more varied).
`performance_profile` is optional and defaults to `fast`. `fast` uses at most
4,096 context tokens and generates up to 384 tokens; `balanced` uses up to
6,144 context tokens and 768 output tokens; `deep` uses up to 8,192 context
tokens and 1,536 output tokens. `OLLAMA_NUM_CTX` caps the context for every
profile. The model stays loaded for `OLLAMA_KEEP_ALIVE` to reduce reload delays.

## Troubleshooting

- **"Ollama is not reachable"** — start Ollama (`ollama serve`) and check
  `OLLAMA_URL`.
- **"Model is not installed"** — run `ollama pull llama3.2:3b` and/or
  `ollama pull qwen3-vl:8b`.
- **Out of memory / very slow image answers** — close other apps, lower
  `OLLAMA_NUM_CTX`, or use a smaller vision model.
- **Upload rejected** — check the extension and the 20 MB limit.
- **Model list looks stale** — refresh the page; the dropdown is rebuilt from
  `/api/models`.

## Project layout

```
app.py               Flask backend: auth, SMTP verification, uploads, Ollama proxy
templates/index.html Main workspace markup
templates/auth.html  Login, signup and email verification screens
static/css/style.css Shared app styling
static/css/auth.css  Login/signup styling
static/js/app.js     Workspace logic and account-scoped browser storage
static/js/auth.js    Sign-in, signup and verification flow
ruas_auth.sqlite3    Local accounts and per-account settings (created at runtime)
requirements.txt     Python dependencies
run.bat              One-click Windows launcher
pull_models.ps1      Downloads the chat and vision models
uploads/             Local attachment store (auto-cleaned)
```
