#!/bin/bash
# Cron/手工入口：加载 ops.env 后跑 Watch（不 Apply）。
set -euo pipefail
set -a
# shellcheck disable=SC1091
. /data/users/aliang/doc/config/cloudigo/ops.env
set +a
cd /data/users/aliang/python/litellm
exec /home/aliang/miniconda3/bin/python3 scripts/watch_rezecyan_pricing.py "$@"
