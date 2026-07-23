#!/usr/bin/env bash
set -euo pipefail
TEST_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd -- "$TEST_DIR/.." && pwd)
INSTALLER="$PROJECT_DIR/scripts/install_test_corpus.sh"
RUNTIME_ROOT=${CORPUS_TEST_RUNTIME_ROOT:-/data/var/file-system-test-corpus-restore-test}
export CORPUS_RUNTIME_ROOT="$RUNTIME_ROOT"
cleanup() {
  "$INSTALLER" uninstall >/dev/null 2>&1 || true
  rm -rf "$RUNTIME_ROOT"
}
trap cleanup EXIT
rm -rf "$RUNTIME_ROOT"
"$INSTALLER" reset
"$INSTALLER" verify
mountpoint -q "$RUNTIME_ROOT/file-system-test-corpus-image"
mountpoint -q "$RUNTIME_ROOT/file-system-test-corpus"
"$INSTALLER" unmount
! mountpoint -q "$RUNTIME_ROOT/file-system-test-corpus-image"
! mountpoint -q "$RUNTIME_ROOT/file-system-test-corpus"
"$INSTALLER" reset
"$INSTALLER" verify
printf 'restore test passed on %s (%s)\n' "$(hostname)" "$(uname -m)"
