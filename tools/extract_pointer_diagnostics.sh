#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <app-log-file>" >&2
  exit 1
fi

log_file="$1"
if [[ ! -f "$log_file" ]]; then
  echo "Log file not found: $log_file" >&2
  exit 1
fi

echo "== source log =="
echo "$log_file"
echo

echo "== OpenGL context lines =="
grep -n "OpenGL context:" "$log_file" || true
echo

echo "== Pointer mapping lines =="
grep -n "Pointer mapping:" "$log_file" || true
echo

echo "== Scaling/Qt related warnings (best effort) =="
grep -niE "dpi|scale|scaling|pixel|wayland|xcb|qpa|screen|cursor|vispy|qt" "$log_file" | tail -n 80 || true
