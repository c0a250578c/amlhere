from http.server import BaseHTTPRequestHandler
import json
import sqlite3
from contextlib import contextmanager
from typing import Generator

DB_PATH = "/tmp/chappy.db"

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

    def do_GET(self):
        try:
            # URLからuser_idを抽出
            path = self.path.split('?')[0]
            parts = path.strip('/').split('/')
            user_id = parts[-1] if len(parts) >= 2 else None

            if not user_id:
                self._send_json({"error": "Missing user_id"}, 400)
                return

            query = self.path.split('?')[1] if '?' in self.path else ''
            params = {}
            for p in query.split('&'):
                if '=' in p:
                    k, v = p.split('=', 1)
                    params[k] = v
            limit = min(int(params.get('limit', 20)), 100)

            with get_db() as conn:
                rows = conn.execute(
                    "SELECT role, message, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()

            result = [{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows]
            self._send_json(result)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
