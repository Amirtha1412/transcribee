#!/usr/bin/env bash

# transcribee 🐝 - YouTube/local video transcription for LLM context
# Usage: transcribee https://www.youtube.com/watch?v=...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# If no arguments provided, show usage
if [ $# -eq 0 ]; then
    echo "🐝 transcribee - YouTube/local video transcription for LLM context"
    echo ""
    echo "Usage: transcribee <url-or-file>"
    echo ""
    echo "Examples:"
    echo "  transcribee https://www.youtube.com/watch?v=MW3t6jP9AOs"
    echo "  transcribee ~/Videos/interview.mp4"
    exit 1
fi


# Pass the first argument (URL) to the script
pnpm exec tsx index.ts "$1"
