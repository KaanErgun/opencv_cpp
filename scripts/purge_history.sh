#!/usr/bin/env bash
#
# purge_history.sh — Rewrite git history to remove leaked credentials, committed
# binaries, and the 12 MB best.onnx model blob from EVERY past commit.
#
# WARNING: This rewrites history. Every commit SHA changes. After running it you
# MUST force-push, and every existing clone becomes invalid and must be re-cloned.
# This script is NOT run automatically — read it, then run it yourself when ready.
#
# The working tree was already cleaned in v2.0; this only scrubs the PAST commits
# still visible via `git log -p`. The leaked passwords must ALSO be rotated at the
# source (cameras / API accounts) — history rewriting alone does not un-leak them.
#
# Prerequisite:
#   brew install git-filter-repo      # or: pip install git-filter-repo
#
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo not found. Install with: brew install git-filter-repo" >&2
  exit 1
fi

echo ">> Tagging pre-v2 state so old module code stays browsable..."
git tag -f pre-v2 || true

echo ">> Removing binary/model blobs from all history..."
git filter-repo --force \
  --invert-paths \
  --path alpr_plate_detection/plate_recognizer \
  --path yolov3_plate_recognition/plate_recognition \
  --path yolov3_plate_recognition/no_yolo_plate_recognition \
  --path yolov8_car_plates_detection/best.onnx

echo ">> Scrubbing leaked credentials from all history..."
cat > /tmp/purge-replacements.txt <<'EOF'
regex:rtsp://[^@\s"']+:[^@\s"']+@==>rtsp://USER:PASS@
alpDADE2==>REDACTED
Password.123==>REDACTED
Password.1234==>REDACTED
test123==>REDACTED
literal:password=test123==>password=REDACTED
alpdade==>REDACTED
EOF
git filter-repo --force --replace-text /tmp/purge-replacements.txt
rm -f /tmp/purge-replacements.txt

echo ""
echo ">> Verifying no known secrets remain in history..."
if git log -p --all | grep -nE 'alpDADE2|Password\.123|test123|alpdade' >/dev/null; then
  echo "!! Secrets STILL present — inspect manually." >&2
  exit 1
fi
echo "   Clean."

echo ""
echo ">> DONE. Next steps (run manually):"
echo "     git remote add origin https://github.com/KaanErgun/opencv_cpp.git   # filter-repo drops the remote"
echo "     git push --force --tags origin main"
echo "     # Then: rotate the leaked camera/API passwords at their source."
