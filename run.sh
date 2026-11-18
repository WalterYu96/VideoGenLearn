#!/usr/bin/env bash
# 本地预览视频生成学习站
# 用法：./serve.sh [端口]     默认端口 8765
set -euo pipefail

PORT="${1:-10000}"
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  echo "未找到 python3，请先安装 Python 3。" >&2
  exit 1
fi

URL="http://localhost:${PORT}"
echo "站点目录：$(pwd)"
echo "访问地址：${URL}"
echo "按 Ctrl+C 停止"

# macOS 下服务起来后自动打开浏览器
if command -v open >/dev/null 2>&1; then
  (sleep 1 && open "$URL") &
fi

# 加 Cache-Control: no-cache，改了 chapters.js / app.js 后刷新就能生效，不会用到浏览器缓存的旧文件
exec python3 -c '
import http.server, sys
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()
http.server.ThreadingHTTPServer(("", int(sys.argv[1])), H).serve_forever()
' "$PORT"
