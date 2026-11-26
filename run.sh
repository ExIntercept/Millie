#!/usr/bin/env bash
# Start Millie. Binds to localhost only (there is no login, so don't expose it).
set -e
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate
exec uvicorn backend.main:app --host 127.0.0.1 --port 8000
