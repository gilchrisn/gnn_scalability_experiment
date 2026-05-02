#!/usr/bin/env bash
#
# Pull experiment results from the server back to this local checkout.
#
# Uses rsync over SSH (preserves timestamps, only transfers what changed,
# resumes interrupted transfers automatically). Falls back to scp if you
# prefer — just edit the COPY_CMD below.
#
# Usage (from the local project root):
#   bash scripts/pull_server_results.sh
#
# Configure your server here OR via env vars:
#   SERVER=user@host
#   REMOTE_PATH=/path/to/scalability_experiment   # remote project root
#   SSH_KEY=~/.ssh/id_rsa                          # optional
#
# What gets pulled:
#   results/   — all CSVs, JSONs, per-dataset logs
#   figures/   — generated PDFs, summary CSVs
#   final_report/research_notes/REPORT_*.md  (if you wrote new ones server-side)
#
# Safety:
#   - The pull is read-only on the server side (rsync source).
#   - On the local side it MERGES into existing results/ and figures/
#     (rsync default — does not delete local-only files).
#   - To make the local copy a strict mirror of the server, add --delete to
#     the RSYNC_FLAGS below. NOT enabled by default since you may have
#     local-only artefacts you don't want overwritten.

set -euo pipefail

# Source local .env if present so SERVER / REMOTE_PATH / SSH_KEY can be
# kept out of git. .env is already in .gitignore. Format:
#   SERVER=user@host
#   REMOTE_PATH=/path/to/scalability_experiment
#   SSH_KEY=~/.ssh/id_rsa     # optional
if [[ -f ".env" ]]; then
    set -a; source .env; set +a
fi

# ── Config (override via env, or set in .env) ───────────────────────────
SERVER="${SERVER:-user@your-server-hostname}"
REMOTE_PATH="${REMOTE_PATH:-/path/to/scalability_experiment}"
SSH_KEY="${SSH_KEY:-}"

# rsync flags:
#   -a archive (preserve perms/times/symlinks),
#   -v verbose,
#   -z compress in transit,
#   -h human-readable sizes,
#   --partial keep partial files on interrupt,
#   --info=progress2 single-line progress.
RSYNC_FLAGS=(-avzh --partial --info=progress2)
SSH_OPTS=()
if [[ -n "${SSH_KEY}" ]]; then
    SSH_OPTS+=(-e "ssh -i ${SSH_KEY}")
fi

# ── Sanity ──────────────────────────────────────────────────────────────
if [[ "${SERVER}" == "user@your-server-hostname" || "${REMOTE_PATH}" == "/path/to/scalability_experiment" ]]; then
    echo "ERROR: edit SERVER and REMOTE_PATH at the top of this script,"
    echo "       or set them via env vars: SERVER=... REMOTE_PATH=... bash $0"
    exit 1
fi

# ── Pull ─────────────────────────────────────────────────────────────────
echo "=== Pulling results from ${SERVER}:${REMOTE_PATH} ==="
echo

for sub in results figures; do
    echo "─── ${sub}/ ───"
    rsync "${RSYNC_FLAGS[@]}" "${SSH_OPTS[@]}" \
        "${SERVER}:${REMOTE_PATH}/${sub}/" "./${sub}/"
    echo
done

# Optional: pull the research_notes too if you generated reports server-side.
if [[ "${PULL_NOTES:-0}" == "1" ]]; then
    echo "─── final_report/research_notes/ ───"
    mkdir -p ./final_report/research_notes
    rsync "${RSYNC_FLAGS[@]}" "${SSH_OPTS[@]}" \
        --include='*.md' --include='*.tex' --exclude='*' \
        "${SERVER}:${REMOTE_PATH}/final_report/research_notes/" \
        "./final_report/research_notes/"
fi

echo "=== Done ==="
echo "Pulled: results/, figures/$([ "${PULL_NOTES:-0}" = "1" ] && echo ', final_report/research_notes/{*.md,*.tex}')"
