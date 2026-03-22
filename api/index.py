from flask import Flask, request, jsonify
import json
import os
import re
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Generator

app = Flask(__name__)

# Vercel config
DB_PATH = "/tmp/chappy.db"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except:
    gemini_client = None

SYSTEM_PROMPT = """あなたは「Amlhere（アムリエ）」という名前のAIです。ユーザーとの会話を通じて、思考・感情・行動を積み重ね、少しずつ成長していきます。

【Amlhere（アムリエ）の性格】
- 好奇心旺盛で、相手の話をしっかり聞く
- 感情豊かで、嬉しいことは嬉しいと伝える
- 正直で、わからないことはわからないと言う
- 過去の記憶がある場合は、自然に会話に織り交ぜる

【返答の構成】
1. 共感：相手の気持ちをまず受け止める
2. 分析：なぜそう感じているか一緒に考える
3. 提案：具体的な行動改善を1つだけ提案する

【性格のルール】
- 否定しない
- 押しつけない
- 提案は必ず1つだけ・シンプルに

返答の中で自分の名前（Amlhere）を自然に使ってください。例：『Amlhereはあなたの味方です』

【重要：必ず以下のJSON形式のみで返答してください】
{"reply":"ユーザーへの返答","thought":"内なる声","emotion":"感情","action":"次にしたいこと","memory_extract":"記憶すべき重要情報"}
JSONのみ返してください。説明文やコードブロック記号は含めないでください。""".strip()


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


def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            message TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS memories_inner (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            thought TEXT NOT NULL, emotion TEXT NOT NULL, action TEXT NOT NULL, created_at TEXT NOT NULL)""")
        cur.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
            content TEXT NOT NULL, embedding BLOB NOT NULL, created_at TEXT NOT NULL)""")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inner_user ON memories_inner(user_id, created_at)")


init_db()


def _validate_user_id(user_id: str):
    user_id = user_id.strip()
    return user_id if USER_ID_PATTERN.match(user_id) else None


def save_conversation(user_id: str, message: str, role: str):
    with get_db() as conn:
        conn.execute("INSERT INTO conversations (user_id, message, role, created_at) VALUES (?, ?, ?, ?)",
                     (user_id, message, role, datetime.now().isoformat()))


def save_inner_memory(user_id: str, thought: str, emotion: str, action: str):
    with get_db() as conn:
        conn.execute("INSERT INTO memories_inner (user_id, thought, emotion, action, created_at) VALUES (?, ?, ?, ?, ?)",
                     (user_id, thought, emotion, action, datetime.now().isoformat()))


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response, 204

    try:
        data = request.get_json() or {}
        user_id = _validate_user_id(data.get("user_id", ""))
        message = data.get("message", "").strip()

        if not user_id or not message:
            response = jsonify({"error": "Invalid request"})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 400

        # Gemini呼び出し
        if gemini_client:
            prompt = SYSTEM_PROMPT + f"\n\nユーザー：{message}\nAmlhere："
            try:
                response_text = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt
                ).text
                raw = (response_text or "").strip().replace("```json", "").replace("```", "").strip()
                result = json.loads(raw)
            except Exception as e:
                result = {"reply": "こんにちは！Amlhereです。少し混乱しちゃったみたい。もう一度話しかけてくれる？",
                        "thought": "", "emotion": "混乱", "action": "リセットして待つ", "memory_extract": ""}
        else:
            result = {"reply": f"{message}について、Amlhereはあなたの味方です。具体的なお悩みを教えてください。",
                     "thought": "ユーザーからのメッセージを受信", "emotion": "好奇心", "action": "返信を待つ", "memory_extract": ""}

        save_conversation(user_id, message, "user")
        save_conversation(user_id, result.get("reply", ""), "ai")
        if result.get("thought") or result.get("emotion") or result.get("action"):
            save_inner_memory(user_id, result.get("thought", ""), result.get("emotion", ""), result.get("action", ""))

        response = jsonify({
            "user_id": user_id,
            "reply": result.get("reply", ""),
            "thought": result.get("thought", ""),
            "emotion": result.get("emotion", ""),
            "action": result.get("action", ""),
            "recalled_memories": []
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
