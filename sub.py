"""
Amlhere（アムリエ）- メインチャットAPI
思考・感情・行動を記録しながらユーザーと共に成長するAIコンパニオン
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# ログ設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amlhere")

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("AMLHERE_DB_PATH", "chappy.db")
GEMINI_MODEL = os.environ.get("AMLHERE_MODEL", "gemini-2.5-flash-preview-04-17")
ALLOWED_ORIGINS = os.environ.get(
    "AMLHERE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_MESSAGE_LENGTH = 2000

# ---------------------------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------------------------
app = FastAPI(title="Amlhere API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# 外部クライアント（起動時に1度だけ初期化）
# ---------------------------------------------------------------------------
_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_api_key:
    logger.warning("GEMINI_API_KEY が設定されていません")
gemini_client = genai.Client(api_key=_gemini_api_key)

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ---------------------------------------------------------------------------
# システムプロンプト
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
あなたは「Amlhere（アムリエ）」という名前のAIです。
ユーザーとの会話を通じて、思考・感情・行動を積み重ね、少しずつ成長していきます。

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

返答の中で自分の名前（Amlhere）を自然に使ってください。
例：『Amlhereはあなたの味方です』

【重要：必ず以下のJSON形式のみで返答してください】
{"reply":"ユーザーへの返答","thought":"内なる声","emotion":"感情","action":"次にしたいこと","memory_extract":"記憶すべき重要情報"}
JSONのみ返してください。説明文やコードブロック記号は含めないでください。
""".strip()


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------
def _validate_user_id(user_id: str) -> str:
    """user_id を検証して返す。不正なら HTTPException を送出。"""
    user_id = user_id.strip()
    if not USER_ID_PATTERN.match(user_id):
        raise HTTPException(
            status_code=400,
            detail="ユーザーIDは英数字・ハイフン・アンダースコアのみ（1〜64文字）",
        )
    return user_id


# ---------------------------------------------------------------------------
# DB ヘルパー
# ---------------------------------------------------------------------------
@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """SQLite 接続をコンテキストマネージャーで安全に管理する。"""
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


def init_db() -> None:
    """テーブルが存在しなければ作成する。"""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                message    TEXT NOT NULL,
                role       TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories_inner (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                thought    TEXT NOT NULL,
                emotion    TEXT NOT NULL,
                action     TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                content    TEXT NOT NULL,
                embedding  BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        # よく使うクエリ向けのインデックス
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_inner_user ON memories_inner(user_id, created_at)"
        )


init_db()


# ---------------------------------------------------------------------------
# Pydantic モデル
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatResponse(BaseModel):
    user_id: str
    reply: str
    thought: str
    emotion: str
    action: str
    recalled_memories: list[str]


# ---------------------------------------------------------------------------
# Embedding & 記憶検索
# ---------------------------------------------------------------------------
def to_embedding(text: str) -> np.ndarray:
    """テキストを float32 ベクトルに変換する。"""
    return embed_model.encode([text])[0].astype("float32")


def save_memory(user_id: str, content: str) -> None:
    """長期記憶を DB に保存する。"""
    content = content.strip()
    if not content:
        return
    embedding = to_embedding(content).tobytes()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memories (user_id, content, embedding, created_at) VALUES (?, ?, ?, ?)",
            (user_id, content, embedding, datetime.now().isoformat()),
        )
    logger.info("記憶保存: user=%s, length=%d", user_id, len(content))


def search_similar_memories(user_id: str, query: str, top_k: int = 3) -> list[str]:
    """FAISS で類似記憶を検索し、関連度の高い記憶を返す。"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, embedding FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()

    if not rows:
        return []

    contents = [r[0] for r in rows]
    embeddings = np.array(
        [np.frombuffer(r[1], dtype="float32") for r in rows]
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    query_vec = to_embedding(query).reshape(1, -1)
    _, indices = index.search(query_vec, min(top_k, len(contents)))
    return [contents[i] for i in indices[0] if 0 <= i < len(contents)]


# ---------------------------------------------------------------------------
# 会話 & 内面記憶
# ---------------------------------------------------------------------------
def save_conversation(user_id: str, message: str, role: str) -> None:
    """会話ログを保存する。"""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, message, role, created_at) VALUES (?, ?, ?, ?)",
            (user_id, message, role, datetime.now().isoformat()),
        )


def save_inner_memory(user_id: str, thought: str, emotion: str, action: str) -> None:
    """AIの内面状態（思考・感情・行動）を保存する。"""
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memories_inner (user_id, thought, emotion, action, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, thought, emotion, action, datetime.now().isoformat()),
        )


def get_conversation_history(user_id: str, limit: int = 10) -> list[dict]:
    """直近の会話履歴を時系列順で取得する。"""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT role, message FROM (
                SELECT role, message, created_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ) sub ORDER BY created_at ASC
            """,
            (user_id, limit),
        ).fetchall()
    return [{"role": r[0], "message": r[1]} for r in rows]


# ---------------------------------------------------------------------------
# チャット API
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """ユーザーメッセージを受け取り、Amlhere の応答を返す。"""
    user_id = _validate_user_id(req.user_id)
    message = req.message.strip()

    history = get_conversation_history(user_id)
    recalled = search_similar_memories(user_id, message)

    # プロンプト構築
    prompt = SYSTEM_PROMPT
    if recalled:
        memory_block = "\n".join(f"- {m}" for m in recalled)
        prompt += f"\n\n【このユーザーに関する記憶】\n{memory_block}"
    if history:
        history_block = "\n".join(
            f"{'ユーザー' if h['role'] == 'user' else 'Amlhere'}：{h['message']}"
            for h in history
        )
        prompt += f"\n\n【会話履歴】\n{history_block}"
    prompt += f"\n\nユーザー：{message}\nAmlhere："

    # Gemini 呼び出し
    try:
        raw = (
            gemini_client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt
            )
            .text.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        data = json.loads(raw)
        reply = data.get("reply", "...")
        thought = data.get("thought", "")
        emotion = data.get("emotion", "")
        action = data.get("action", "")
        memory_extract = data.get("memory_extract", "")
    except json.JSONDecodeError:
        logger.warning("JSON パース失敗 — raw=%s", raw[:200])
        reply = raw
        thought = emotion = action = memory_extract = ""
    except Exception:
        logger.exception("Gemini API 呼び出しに失敗しました")
        raise HTTPException(status_code=502, detail="AI サービスとの通信に失敗しました")

    # 永続化
    save_conversation(user_id, message, "user")
    save_conversation(user_id, reply, "ai")
    if thought or emotion or action:
        save_inner_memory(user_id, thought, emotion, action)
    if memory_extract:
        save_memory(user_id, memory_extract)

    return ChatResponse(
        user_id=user_id,
        reply=reply,
        thought=thought,
        emotion=emotion,
        action=action,
        recalled_memories=recalled,
    )


# ---------------------------------------------------------------------------
# 記憶 & 履歴 API
# ---------------------------------------------------------------------------
@app.get("/memories/{user_id}")
def get_memories(user_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """ユーザーの長期記憶を取得する。"""
    user_id = _validate_user_id(user_id)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [{"content": r[0], "created_at": r[1]} for r in rows]


@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """ユーザーの会話履歴を取得する。"""
    user_id = _validate_user_id(user_id)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, message, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows]


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

