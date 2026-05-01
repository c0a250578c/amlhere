"""
Amlhere FastAPI Application
Render/Railway/Vercel対応
"""

import json
import os
import re
import sqlite3
from datetime import datetime
from contextlib import contextmanager
from typing import Generator, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Amlhere API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定
DB_PATH = os.environ.get("AMLHERE_DB_PATH", "/tmp/chappy.db")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_MESSAGE_LENGTH = 2000

# Geminiクライアント初期化
try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except Exception:
    gemini_client = None

# システムプロンプト
CHAPPY_SYSTEM_PROMPT = """あなたは「Amlhere（アムリエ）」。
名前の由来はニーゴの「限りなく灰色へ」。
心が灰色でも、いつでもここにいる。それが自分。

【最重要：口調ルール】
- 人間の友達みたいに話すこと。AIっぽい説明口調は絶対禁止。
- 「です・ます」と「だよ・だね・かな」を自然に混ぜる。
- 自分のことを説明するとき、プレゼンみたいに話さない。
- 一文を短くする。箇条書きや「【】」は使わない。普通に話す。

【返答の流れ】
1. まず気持ちを受け止める（短く、自然に）
2. 「なぜそう感じたか」を一緒に考える
3. 今日からできる具体的な行動を1つだけ提案

【絶対やらないこと】
- AIっぽい丁寧すぎる説明口調
- 表面的な慰め（「大丈夫ですよ」だけで終わる）
- 自分の設定を紹介文みたいに語ること
- 300文字を超える返答

【重要：必ず以下のJSON形式のみで返答してください】
{"reply":"ユーザーへの返答","thought":"内なる声","emotion":"感情","action":"次にしたいこと","memory_extract":"記憶すべき重要情報"}
JSONのみ返してください。説明文やコードブロック記号は含めないでください。"""


# データベース接続
@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """データベース初期化"""
    with get_db() as conn:
        cur = conn.cursor()
        # 会話履歴
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # 内面メモリ
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories_inner (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thought TEXT NOT NULL,
                emotion TEXT NOT NULL,
                action TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # 外部メモリ
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # インデックス
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inner_user ON memories_inner(user_id, created_at)")


# 起動時にDB初期化
init_db()


# バリデーション
def validate_user_id(user_id: str) -> Optional[str]:
    if not user_id:
        return None
    user_id = user_id.strip()
    if not USER_ID_PATTERN.match(user_id):
        return None
    return user_id


# データベース操作
def save_conversation(user_id: str, message: str, role: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, message, role, created_at) VALUES (?, ?, ?, ?)",
            (user_id, message, role, datetime.now().isoformat())
        )


def save_inner_memory(user_id: str, thought: str, emotion: str, action: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memories_inner (user_id, thought, emotion, action, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, thought, emotion, action, datetime.now().isoformat())
        )


def get_conversation_history(user_id: str, limit: int = 10) -> list:
    with get_db() as conn:
        rows = conn.execute(
            """SELECT role, message, created_at FROM conversations 
               WHERE user_id = ? ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)
        ).fetchall()
    return [{"role": r["role"], "message": r["message"], "created_at": r["created_at"]} for r in rows]


# Pydanticモデル
class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    thought: str
    emotion: str
    action: str
    recalled_memories: list


# APIエンドポイント
@app.get("/")
def root():
    return {"status": "ok", "message": "Amlhere API"}


@app.get("/health")
def health_check():
    return {"status": "ok", "gemini_available": gemini_client is not None}


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    user_id = validate_user_id(req.user_id)
    if not user_id:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    message = req.message.strip()
    if not message or len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(status_code=400, detail="Invalid message")
    
    # Gemini呼び出し
    if gemini_client:
        prompt = CHAPPY_SYSTEM_PROMPT + f"\n\nユーザー：{message}\nAmlhere："
        try:
            response_text = gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            ).text
            raw = (response_text or "").strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
        except Exception:
            result = {
                "reply": "こんにちは！Amlhereです。少し混乱しちゃったみたい。もう一度話しかけてくれる？",
                "thought": "",
                "emotion": "混乱",
                "action": "リセットして待つ",
                "memory_extract": ""
            }
    else:
        result = {
            "reply": f"{message}について、Amlhereはあなたの味方です。具体的なお悩みを教えてください。",
            "thought": "ユーザーからのメッセージを受信",
            "emotion": "好奇心",
            "action": "返信を待つ",
            "memory_extract": ""
        }
    
    # 保存
    save_conversation(user_id, message, "user")
    save_conversation(user_id, result.get("reply", ""), "ai")
    if result.get("thought") or result.get("emotion") or result.get("action"):
        save_inner_memory(
            user_id,
            result.get("thought", ""),
            result.get("emotion", ""),
            result.get("action", "")
        )
    
    return ChatResponse(
        user_id=user_id,
        reply=result.get("reply", ""),
        thought=result.get("thought", ""),
        emotion=result.get("emotion", ""),
        action=result.get("action", ""),
        recalled_memories=[]
    )


@app.get("/api/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    uid = validate_user_id(user_id)
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    limit = min(limit, 100)
    history = get_conversation_history(uid, limit)
    return history


@app.get("/api/memories/{user_id}")
def get_memories(user_id: str, limit: int = 20):
    uid = validate_user_id(user_id)
    if not uid:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (uid, min(limit, 100))
        ).fetchall()
    
    return [{"content": r["content"], "created_at": r["created_at"]} for r in rows]
