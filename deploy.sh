#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────
# SSI-ENN BESS — One-Command Deploy
# ───────────────────────────────────────────────────────────────────
# Usage:
#   ./deploy.sh                  # Full pipeline + push
#   ./deploy.sh --skip-pipeline  # Push current files only
#   ./deploy.sh --dry-run        # Pipeline dry-run, no push
#
# Prerequisites:
#   - Python 3.9+ with numpy, requests
#   - git remote 'origin' configured with push access
# ───────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No colour

# ── Parse arguments ──
SKIP_PIPELINE=false
DRY_RUN=false
COMMIT_MSG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --skip-pipeline) SKIP_PIPELINE=true; shift ;;
    --dry-run)       DRY_RUN=true; shift ;;
    --message|-m)    COMMIT_MSG="$2"; shift 2 ;;
    *)               echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SSI-ENN BESS — Deploy Pipeline${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ── Step 1: Run ingestion pipeline ──
if [ "$SKIP_PIPELINE" = false ]; then
  echo -e "${YELLOW}[1/4]${NC} Running ingestion pipeline..."
  echo ""

  PIPELINE_ARGS=""
  if [ "$DRY_RUN" = true ]; then
    PIPELINE_ARGS="--dry-run"
  fi

  python3 -m pipeline.run_ingestion $PIPELINE_ARGS

  if [ $? -ne 0 ]; then
    echo -e "${RED}Pipeline failed. Aborting deploy.${NC}"
    exit 1
  fi
  echo ""
else
  echo -e "${YELLOW}[1/4]${NC} Skipping pipeline (--skip-pipeline)"
  echo ""
fi

# ── Step 2: Read version ──
VERSION_FILE="pipeline/version.txt"
if [ -f "$VERSION_FILE" ]; then
  VERSION=$(cat "$VERSION_FILE")
  echo -e "${YELLOW}[2/4]${NC} Current version: ${GREEN}v${VERSION}${NC}"
else
  VERSION="0"
  echo -e "${YELLOW}[2/4]${NC} No version.txt found, using v0"
fi
echo ""

# ── Step 3: Verify outputs ──
echo -e "${YELLOW}[3/4]${NC} Verifying outputs..."

if [ ! -f "data.json" ]; then
  echo -e "${RED}  MISSING: data.json${NC}"
  exit 1
fi

RECORD_COUNT=$(python3 -c "import json; print(len(json.load(open('data.json'))))" 2>/dev/null || echo "0")
DATA_SIZE=$(du -h data.json | cut -f1)
echo -e "  data.json: ${GREEN}${RECORD_COUNT} records${NC} (${DATA_SIZE})"

if [ -f "pipeline/audit_report.json" ]; then
  echo -e "  audit_report.json: ${GREEN}present${NC}"
else
  echo -e "  audit_report.json: ${YELLOW}missing (non-critical)${NC}"
fi

# Check HTML cache-busters match version
MISMATCHED=0
for html_file in *.html; do
  if grep -qP "\?v=\d+" "$html_file" 2>/dev/null; then
    OLD_REFS=$(grep -oP '\?v=\d+' "$html_file" | sort -u | grep -v "?v=${VERSION}" | wc -l)
    if [ "$OLD_REFS" -gt 0 ]; then
      echo -e "  ${YELLOW}WARNING:${NC} ${html_file} has stale cache-busters"
      MISMATCHED=$((MISMATCHED + 1))
    fi
  fi
done

if [ "$MISMATCHED" -eq 0 ]; then
  echo -e "  HTML cache-busters: ${GREEN}all match v${VERSION}${NC}"
fi
echo ""

# ── Step 4: Git commit & push ──
if [ "$DRY_RUN" = true ]; then
  echo -e "${YELLOW}[4/4]${NC} DRY RUN — skipping git operations"
  echo ""
  echo -e "${GREEN}Dry run complete.${NC}"
  exit 0
fi

echo -e "${YELLOW}[4/4]${NC} Committing and pushing..."

# Check for changes
if git diff --quiet && git diff --cached --quiet; then
  echo -e "  No changes to commit."
else
  # Stage data files + HTML + pipeline
  git add data.json
  git add pipeline/version.txt pipeline/audit_report.json 2>/dev/null || true
  git add *.html
  git add pipeline/*.py
  git add deploy.sh 2>/dev/null || true

  # Build commit message
  if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Deploy v${VERSION}: ${RECORD_COUNT} substations, pipeline update"
  fi

  git commit -m "$COMMIT_MSG"
  echo -e "  Committed: ${GREEN}${COMMIT_MSG}${NC}"
fi

# Push
echo -e "  Pushing to origin/main..."
git push origin main

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deploy complete — v${VERSION}${NC}"
echo -e "${GREEN}  ${RECORD_COUNT} substations live at:${NC}"
echo -e "${GREEN}  https://ikengatest.github.io/ssi-bess-private/${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
