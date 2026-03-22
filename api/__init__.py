"""
Amlhere - Vercel Serverless API
共用モジュール
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from html import escape as html_escape
from typing import Any, Generator

import numpy as np
from dotenv import load_dotenv
from google import genai

# Vercelは/tmpのみ書き込み可能
DB_PATH = "/tmp/chappy.db"
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_DAYS = 365

load_dotenv()

# Geminiクライアント
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# システムプロンプト
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


# バリデーション
def _validate_user_id(user_id: str) -> str:
    user_id = user_id.strip()
    if not USER_ID_PATTERN.match(user_id):
        return None
    return user_id


# DB ヘルパー
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                message TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inner_user ON memories_inner(user_id, created_at)")


# 起動時に初期化
init_db()
