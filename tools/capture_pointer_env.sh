#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-file>" >&2
  exit 1
fi

out_file="$1"
out_dir="$(dirname "$out_file")"
mkdir -p "$out_dir"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
  echo "python_version=$(python -V 2>&1 || echo n/a)"
  echo "display=${DISPLAY-}"
  echo "wayland_display=${WAYLAND_DISPLAY-}"
  echo "xdg_session_type=${XDG_SESSION_TYPE-}"
  echo "qt_qpa_platform=${QT_QPA_PLATFORM-}"
  echo "qt_scale_factor=${QT_SCALE_FACTOR-}"
  echo "qt_auto_screen_scale_factor=${QT_AUTO_SCREEN_SCALE_FACTOR-}"
  echo "qt_screen_scale_factors=${QT_SCREEN_SCALE_FACTORS-}"
  echo "gdk_scale=${GDK_SCALE-}"
  echo "xcursor_size=${XCURSOR_SIZE-}"
  echo "xra_use_legacy_pointer_scale=${XRA_USE_LEGACY_POINTER_SCALE-}"
} > "$out_file"

echo "Wrote pointer environment snapshot: $out_file"
