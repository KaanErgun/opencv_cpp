#!/usr/bin/env bash
#
# check.sh — the single local quality gate. There is NO CI (project rule); run
# this before every push.
#
#   ./scripts/check.sh
#
# Steps: configure + build (Release), run unit tests, verify formatting, and
# grep for accidentally-committed credentials.
#
set -euo pipefail
cd "$(dirname "$0")/.."

BUILD_DIR="build/check"
OPENCV_DIR="${OpenCV_DIR:-/usr/local/Cellar/opencv/5.0.0/lib/cmake/opencv5}"

# The dev machine is arm64 but Homebrew OpenCV here is x86_64; keep them aligned.
ARCH_FLAG=""
if [[ "$(uname -m)" == "arm64" && -d /usr/local/Cellar/opencv ]]; then
  ARCH_FLAG="-DCMAKE_OSX_ARCHITECTURES=x86_64"
fi

echo ">> [1/4] Configure + build (Release)..."
cmake -S . -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$OPENCV_DIR" \
  $ARCH_FLAG >/dev/null
cmake --build "$BUILD_DIR" >/dev/null
echo "   build ok"

echo ">> [2/4] Unit tests..."
ctest --test-dir "$BUILD_DIR" --output-on-failure

echo ">> [3/4] clang-format check..."
FILES=$(git ls-files '*.cpp' '*.hpp' | grep -v thirdparty || true)
if [[ -n "$FILES" ]]; then
  clang-format --dry-run --Werror $FILES
  echo "   format ok"
fi

echo ">> [4/4] Credential scan..."
# Flag real-looking rtsp credentials and password= assignments, but allow the
# documented placeholders (USER:PASS@CAMERA_IP, password=REDACTED).
if git ls-files '*.cpp' '*.hpp' '*.json' | xargs grep -nE \
    'rtsp://[^"]*:[^"]*@|password=[^"&]*[A-Za-z0-9]' 2>/dev/null \
    | grep -viE 'USER:PASS@CAMERA_IP|user:pass@host|password=REDACTED|password=PASS'; then
  echo "!! Possible credential(s) found above — do not commit." >&2
  exit 1
fi
echo "   clean"

echo ""
echo "All checks passed."
