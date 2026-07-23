#!/usr/bin/env bash
# audit_secrets.sh — refuse to ship a secret.
#
# Scans BOTH the full commit history (`git log -p --all` — every version of
# every tracked file ever committed, on every branch) AND the current tracked
# worktree for credential-shaped strings. Any hit exits 1 so CI or a release
# can block on it.
#
# `.env` is where real secrets live locally (the RunPod API key). It MUST stay
# untracked and gitignored: this script asserts that loudly and never reads its
# contents. Because the worktree scan uses `git grep` (tracked files only, and
# it honours .gitignore), .env and .venv are excluded automatically.
#
# Patterns (extend as new providers appear):
#   rpa_…                RunPod API key
#   AKIA…                AWS access key id
#   ghp_…                GitHub personal access token
#   -----BEGIN … KEY     PEM private-key header
#   sk-…                 OpenAI-style secret key
#
# Note: the alternation below is written with regex metacharacters ([, {) right
# after each prefix, so this script does not match ITS OWN patterns.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 2

PATTERNS='rpa_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|-----BEGIN[A-Z ]*PRIVATE KEY|sk-[A-Za-z0-9]{20,}'

status=0
hist_hits="$(mktemp)"
tree_hits="$(mktemp)"
trap 'rm -f "$hist_hits" "$tree_hits"' EXIT

echo "== .env must never be tracked =="
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "FAIL: .env is TRACKED by git — secrets belong only in the gitignored .env"
  status=1
else
  echo "OK: .env is not tracked"
fi

echo
echo "== worktree scan (tracked files) =="
# git grep exits 0 when it finds a match, 1 when it finds none.
if git grep -nIE "$PATTERNS" -- . ':!scripts/audit_secrets.sh' >"$tree_hits" 2>/dev/null; then
  echo "FAIL: secret-shaped strings in tracked worktree files:"
  cat "$tree_hits"
  status=1
else
  echo "OK: no secret patterns in tracked worktree files"
fi

echo
echo "== history scan (git log -p --all) =="
if git log -p --all 2>/dev/null | grep -nIE "$PATTERNS" >"$hist_hits"; then
  echo "FAIL: secret-shaped strings in commit history:"
  cat "$hist_hits"
  status=1
else
  echo "OK: no secret patterns in commit history"
fi

echo
if [ "$status" -eq 0 ]; then
  echo "AUDIT CLEAN"
else
  echo "AUDIT FAILED — resolve the findings above before releasing."
fi
exit "$status"
