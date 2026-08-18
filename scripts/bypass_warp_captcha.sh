#!/usr/bin/env bash
# Route Aliyun captcha traffic (init/verify + AliyunCaptcha.js) OUTSIDE the
# Cloudflare WARP tunnel, so chat.z.ai's captcha requests egress from the home
# residential IP instead of Cloudflare datacenter IPs (root cause of Aliyun F001).
#
# Requires root:  sudo bash scripts/bypass_warp_captcha.sh
#
# Idempotent: re-running adds missing rules, removes stale ones (IPs that no
# longer resolve), and never touches chat.z.ai (146.19.236.205) which stays on
# WARP. Rules are runtime-only (lost on reboot); re-run after reboot.

set -euo pipefail

HOSTS=(
  "no8xfe.captcha-open-southeast.aliyuncs.com"
  "no8xfe-verify.captcha-open-southeast.aliyuncs.com"
  "o.alicdn.com"
)

PREF=1000

if [[ $EUID -ne 0 ]]; then
  echo "error: run as root (sudo bash $0)" >&2
  exit 1
fi

# --- undo: remove every pref-1000 rule we installed ---
if [[ "${1:-}" == "--undo" ]]; then
  echo "== removing pref $PREF rules =="
  for ip in $(ip rule show | awk -v pref="$PREF" '$1 ~ "^" pref ":" {print $5}'); do
    ip rule del to "$ip" lookup main pref "$PREF" 2>/dev/null || true
    echo "  removed: $ip"
  done
  echo "== remaining pref $PREF rules =="
  ip rule show | grep "pref $PREF" || echo "  (none)"
  ip route flush cache 2>/dev/null || true
  echo "== captcha routes (should be back on WARP) =="
  ip route get 47.236.70.49
  exit 0
fi

# --- resolve current IPv4 set ---
declare -A TARGETS
for h in "${HOSTS[@]}"; do
  while IFS= read -r ip; do
    [[ -n "$ip" ]] && TARGETS["$ip"]=1
  done < <(getent ahostsv4 "$h" | awk '{print $1}')
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  echo "error: could not resolve any IPs for: ${HOSTS[*]}" >&2
  exit 1
fi

echo "== target IPs (via main/home table) =="
for ip in "${!TARGETS[@]}"; do
  echo "  $ip"
done

# --- add missing rules ---
for ip in "${!TARGETS[@]}"; do
  if ip rule show | grep -q "to $ip lookup main pref $PREF"; then
    echo "== rule already present: $ip"
  else
    ip rule add to "$ip" lookup main pref "$PREF"
    echo "== added rule: to $ip lookup main pref $PREF"
  fi
done

# --- remove stale rules (resolved IPs no longer targeted) ---
for line in $(ip rule show | awk -v pref="$PREF" '$0 ~ "pref " pref " " && $1=="to" {print $2}'); do
  if [[ -z "${TARGETS[$line]:-}" ]]; then
    ip rule del to "$line" lookup main pref "$PREF" 2>/dev/null || true
    echo "== removed stale rule: $line"
  fi
done

echo
echo "== final ip rule (captcha section) =="
ip rule show | grep "lookup main pref $PREF" || echo "  (none)"
echo
echo "== route sanity =="
ip route get "${!TARGETS[@]}" 2>/dev/null || true
echo
echo "== chat.z.ai must stay on WARP (untouched) =="
ip route get 146.19.236.205