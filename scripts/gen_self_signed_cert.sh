#!/usr/bin/env bash
# Generate a self-signed TLS cert for LAN-internal use.
set -euo pipefail

CERT_DIR="${CERT_DIR:-$(cd "$(dirname "$0")/.." && pwd)/volumes/certs}"
CERT_CN="${CERT_CN:-orgchat.lan}"
CERT_DAYS="${CERT_DAYS:-3650}"

mkdir -p "$CERT_DIR"
KEY="$CERT_DIR/orgchat.key"
CRT="$CERT_DIR/orgchat.crt"

if [[ -f "$KEY" && -f "$CRT" ]]; then
  echo "cert already present at $CERT_DIR — leaving in place"
  exit 0
fi

openssl req -x509 -nodes -newkey rsa:4096 \
  -keyout "$KEY" \
  -out "$CRT" \
  -days "$CERT_DAYS" \
  -subj "/CN=$CERT_CN" \
  -addext "subjectAltName=DNS:$CERT_CN,DNS:localhost,IP:127.0.0.1"

chmod 600 "$KEY"
chmod 644 "$CRT"
echo "wrote $CRT and $KEY (CN=$CERT_CN, days=$CERT_DAYS)"
