# The Agents Playbook

A collection of agent recipes and best practices.

## Setup

```bash
uv sync
```

## Environment

Copy the example env file and fill in your API key:

```bash
cp .env.example .env
```

Edit `.env` with your OpenRouter key:

```
OPENAI_API_KEY=sk-or-v1-your-actual-key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Running Scripts

Run from the project root:

```bash
uv run python 01-basic-calls/02-basic-chat.py
```

Or activate the venv and use `python` directly:

```bash
source .venv/bin/activate
python 01-basic-calls/02-basic-chat.py
```