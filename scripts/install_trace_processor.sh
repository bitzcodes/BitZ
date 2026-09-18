#!/usr/bin/env bash
# Install the native Perfetto processor inside this checkout. No sudo or Python.
# Version and checksums: https://github.com/google/perfetto/blob/main/tools/trace_processor
set -euo pipefail

if [[ $# -gt 0 ]]; then
  if [[ $# -eq 1 && ( $1 == --help || $1 == -h ) ]]; then
    echo "Usage: bash scripts/install_trace_processor.sh"
    echo "Installs Perfetto v58.2 into .tools/perfetto/ and prints its absolute path."
    exit 0
  fi
  echo "Usage: bash scripts/install_trace_processor.sh" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
install_dir="$repo_root/.tools/perfetto"
processor="$install_dir/trace_processor_shell"
version=v58.2

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)
    platform=mac-arm64
    checksum=d29864d1ba3b36855527bb1b0ca3aa7f703cdce338b9680bb922c5c151b358fa
    ;;
  Darwin/x86_64)
    platform=mac-amd64
    checksum=3927a2767eadd140db3ff4fe0dfbf1bde35c1f56501149cd367f5cee898bef27
    ;;
  Linux/x86_64)
    platform=linux-amd64
    checksum=58042408e6cc861fb1a731c26bb082dc222285561eaa4e12a48a8b2b90dca7b9
    ;;
  Linux/aarch64|Linux/arm64)
    platform=linux-arm64
    checksum=0e6e0c5452c505c8d46fe472fd196a0d17d963460727e2ce2013b02aa1309555
    ;;
  Linux/armv6l|Linux/armv7l|Linux/armv8l)
    platform=linux-arm
    checksum=09683fed93a3452d9dac1f5165a592ec9fec82fba90be2c5dd764c8f9a449e33
    ;;
  *)
    echo "Unsupported platform: $(uname -s)/$(uname -m). Use a native Perfetto build." >&2
    exit 1
    ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
  hash_command=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  hash_command=(shasum -a 256)
else
  echo "Install sha256sum or shasum to verify the download." >&2
  exit 1
fi

matches_checksum() {
  local actual
  actual="$("${hash_command[@]}" "$1")"
  [[ ${actual%% *} == "$checksum" ]]
}

if [[ ! -f "$processor" ]] || ! matches_checksum "$processor"; then
  command -v curl >/dev/null 2>&1 || { echo "Install curl first." >&2; exit 1; }
  mkdir -p "$install_dir"
  download="$(mktemp "$install_dir/.trace_processor_shell.XXXXXX")"
  trap 'rm -f -- "$download"' EXIT
  echo "Downloading Perfetto $version ($platform)..." >&2
  curl --fail --location --show-error --silent --retry 3 \
    "https://commondatastorage.googleapis.com/perfetto-luci-artifacts/$version/$platform/trace_processor_shell" \
    --output "$download"
  if ! matches_checksum "$download"; then
    echo "Perfetto checksum mismatch; the existing installation was not replaced." >&2
    exit 1
  fi
  chmod +x "$download"
  mv -f -- "$download" "$processor"
fi

chmod +x "$processor"
"$processor" --version >&2
printf '%s\n' "$processor"
