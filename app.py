"""Serve the question page and its local JSON API."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json

from build import DATA, ROOT
from search import Search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    index = Search()
    manifest = json.loads((DATA / "manifest.json").read_text())
    metadata = {key: sum(r[key] for r in manifest) for key in ("pages", "chunks", "table_rows")}
    metadata["companies"] = len(manifest)

    class Handler(BaseHTTPRequestHandler):
        def send(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, obj, status=200):
            self.send(json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            if self.path == "/":
                self.send((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/corpus":
                self.send_json(metadata)
            else:
                self.send_json({"error": "页面不存在"}, 404)

        def do_POST(self):
            if self.path != "/api/ask":
                self.send_json({"error": "接口不存在"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 8192:
                    raise ValueError
                payload = json.loads(self.rfile.read(size))
                question = payload.get("question", "") if isinstance(payload, dict) else ""
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 450:
                    raise ValueError
            except (ValueError, UnicodeDecodeError):
                self.send_json({"error": "请输入 1–450 字的问题"}, 400)
                return
            self.send_json(index.ask(question.strip()))

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"财报问答运行于 http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
