from flask import Flask, request, jsonify
import sqlite3
from contextlib import contextmanager
from typing import Generator

app = Flask(__name__)
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


@app.route('/api/memories/<user_id>', methods=['GET', 'OPTIONS'])
def get_memories(user_id):
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 204

    try:
        limit = min(int(request.args.get('limit', 20)), 100)

        with get_db() as conn:
            rows = conn.execute(
                "SELECT content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()

        result = [{"content": r[0], "created_at": r[1]} for r in rows]

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
