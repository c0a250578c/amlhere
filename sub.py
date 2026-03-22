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
from datetime import datetime, timedelta
from html import escape as html_escape
from pathlib import Path
from typing import Any, Generator

import faiss  # type: ignore[import-untyped]
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field
import hashlib

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
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("AMLHERE_DB_PATH", "/tmp/chappy.db")
GEMINI_MODEL = os.environ.get("AMLHERE_MODEL", "gemini-2.5-flash-preview-04-17")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_MESSAGE_LENGTH = 2000
MAX_DAYS = 365

# ---------------------------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------------------------
app = FastAPI(title="Amlhere API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# ---------------------------------------------------------------------------
# 外部クライアント（起動時に1度だけ初期化）
# ---------------------------------------------------------------------------
_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_api_key:
    logger.warning("GEMINI_API_KEY が設定されていません")
gemini_client = genai.Client(api_key=_gemini_api_key)


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
    """テキストをハッシュベースの簡易ベクトルに変換。"""
    hash_val = hashlib.sha256(text.encode()).digest()
    vec = np.frombuffer(hash_val, dtype=np.uint8).astype("float32")
    return vec / np.linalg.norm(vec)


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
    index = faiss.IndexFlatL2(dim)  # type: ignore[no-any-unimported]
    index.add(embeddings)  # type: ignore[no-any-unimported]

    query_vec = to_embedding(query).reshape(1, -1)
    _, idx_array = index.search(query_vec, min(top_k, len(contents)))  # type: ignore[no-any-unimported]
    result_indices: list[int] = [int(x) for x in idx_array[0]]  # type: ignore[no-any-unimported]
    return [contents[i] for i in result_indices if 0 <= i < len(contents)]


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


def get_conversation_history(user_id: str, limit: int = 10) -> list[dict[str, str]]:
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
    raw = ""
    try:
        response_text = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        ).text
        raw = (
            (response_text or "")
            .strip()
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
        logger.warning("JSON パース失敗 — raw=%s", raw[:200] if raw else "(empty)")
        reply = raw or "..."
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


# ===========================================================================
# 感情分析 & ダッシュボード（旧 analytics.py を統合）
# ===========================================================================
EMOTION_SCORE_MAP: dict[str, dict[str, int]] = {
    "嬉しい": {"stress": 1, "motivation": 9},
    "楽しい": {"stress": 1, "motivation": 9},
    "わくわく": {"stress": 2, "motivation": 8},
    "好奇心": {"stress": 2, "motivation": 8},
    "穏やか": {"stress": 2, "motivation": 6},
    "普通":   {"stress": 5, "motivation": 5},
    "不安":   {"stress": 7, "motivation": 3},
    "悲しい": {"stress": 7, "motivation": 2},
    "疲れ":   {"stress": 8, "motivation": 2},
    "辛い":   {"stress": 9, "motivation": 1},
    "怒り":   {"stress": 9, "motivation": 3},
}

DEFAULT_SCORES = {"stress": 5, "motivation": 5}


def emotion_to_score(emotion: str) -> dict[str, int]:
    """感情テキストをストレス・モチベスコアに変換する。"""
    for key, scores in EMOTION_SCORE_MAP.items():
        if key in emotion:
            return scores
    return DEFAULT_SCORES


def get_emotion_logs(user_id: str, days: int = 7) -> list[dict[str, Any]]:
    """過去 N 日分の感情ログを取得する。"""
    since = (datetime.now() - timedelta(days=days)).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT emotion, created_at FROM memories_inner
            WHERE user_id = ? AND created_at >= ?
            ORDER BY created_at ASC
            """,
            (user_id, since),
        ).fetchall()
    return [
        {
            "date": created_at[:10],
            "emotion": emotion,
            **emotion_to_score(emotion),
        }
        for emotion, created_at in rows
    ]


def aggregate_by_day(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """日ごとにストレス・モチベの平均を計算する。"""
    daily: dict[str, dict[str, list[int]]] = {}
    for log in logs:
        date = log["date"]
        if date not in daily:
            daily[date] = {"stress": [], "motivation": []}
        daily[date]["stress"].append(log["stress"])
        daily[date]["motivation"].append(log["motivation"])
    return [
        {
            "date": date,
            "avg_stress": round(sum(d["stress"]) / len(d["stress"]), 1),
            "avg_motivation": round(sum(d["motivation"]) / len(d["motivation"]), 1),
            "count": len(d["stress"]),
        }
        for date, d in sorted(daily.items())
    ]


def _generate_feedback(daily: list[dict[str, Any]], days: int) -> str:
    """Gemini で感情データに基づくフィードバックを生成する。"""
    summary = json.dumps(daily, ensure_ascii=False)
    prompt = (
        f"以下は過去{days}日間のユーザーの感情データです"
        f"（ストレス10段階・モチベ10段階）。\n\n{summary}\n\n"
        "Amlhereとして優しく共感的に2〜3文でフィードバックしてください。日本語で。"
    )
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return (response.text or "").strip()
    except Exception:
        logger.exception("フィードバック生成に失敗しました")
        return "フィードバックを生成できませんでした。しばらくしてからお試しください。"


@app.get("/analytics/{user_id}")
def get_analytics(
    user_id: str,
    days: int = Query(default=7, ge=1, le=MAX_DAYS),
) -> dict[str, Any]:
    """日ごとの感情データを返す。"""
    user_id = _validate_user_id(user_id)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)
    return {
        "user_id": user_id,
        "period_days": days,
        "daily": daily,
        "total_records": len(logs),
    }


@app.get("/feedback/{user_id}")
def get_feedback(
    user_id: str,
    days: int = Query(default=7, ge=1, le=MAX_DAYS),
) -> dict[str, Any]:
    """AI が感情ログを分析してフィードバックを生成する。"""
    user_id = _validate_user_id(user_id)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)
    if not daily:
        return {"feedback": "まだデータが足りないよ。もう少し話しかけてね！"}
    feedback = _generate_feedback(daily, days)
    return {
        "user_id": user_id,
        "period_days": days,
        "feedback": feedback,
        "daily": daily,
    }


@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
def get_dashboard(
    user_id: str,
    days: int = Query(default=7, ge=1, le=MAX_DAYS),
):
    """グラフ + AI フィードバックのダッシュボード画面を返す。"""
    user_id = _validate_user_id(user_id)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)
    if not daily:
        return HTMLResponse(
            "<h2 style='text-align:center;margin-top:40px;color:#eee;font-family:sans-serif'>"
            "データがまだありません。Amlhereと話しかけてみてください！</h2>"
        )
    labels = [d["date"] for d in daily]
    stress_data = [d["avg_stress"] for d in daily]
    motivation_data = [d["avg_motivation"] for d in daily]
    avg_stress = round(sum(stress_data) / len(stress_data), 1)
    avg_motivation = round(sum(motivation_data) / len(motivation_data), 1)
    feedback = html_escape(_generate_feedback(daily, days))
    labels_json = json.dumps(labels)
    stress_json = json.dumps(stress_data)
    motivation_json = json.dumps(motivation_data)
    safe_user_id = html_escape(user_id)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Amlhere Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            min-height: 100vh; color: #eee; padding: 30px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{
            text-align: center; font-size: 2rem; margin-bottom: 8px;
            background: linear-gradient(90deg, #a8edea, #fed6e3);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #aaa; margin-bottom: 30px; font-size: 0.9rem; }}
        .card {{
            background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px; padding: 24px; margin-bottom: 20px; backdrop-filter: blur(10px);
        }}
        .card h2 {{ font-size: 1rem; color: #a8edea; margin-bottom: 16px; letter-spacing: 1px; }}
        .feedback {{ font-size: 1.1rem; line-height: 1.8; color: #eee; border-left: 3px solid #a8edea; padding-left: 16px; }}
        .chart-container {{ position: relative; height: 250px; }}
        .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
        .stat {{ flex: 1; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #a8edea; }}
        .stat-label {{ font-size: 0.8rem; color: #aaa; margin-top: 4px; }}
        @media (max-width: 600px) {{ body {{ padding: 16px; }} .stats {{ flex-direction: column; }} }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Amlhere</h1>
        <p class="subtitle">過去{days}日間の感情レポート / {safe_user_id}</p>
        <div class="stats">
            <div class="stat"><div class="stat-value">{avg_stress}</div><div class="stat-label">平均ストレス</div></div>
            <div class="stat"><div class="stat-value">{avg_motivation}</div><div class="stat-label">平均モチベ</div></div>
            <div class="stat"><div class="stat-value">{len(logs)}</div><div class="stat-label">総記録数</div></div>
        </div>
        <div class="card"><h2>Amlhereからのフィードバック</h2><p class="feedback">{feedback}</p></div>
        <div class="card"><h2>感情の推移</h2><div class="chart-container"><canvas id="chart"></canvas></div></div>
    </div>
    <script>
        new Chart(document.getElementById('chart').getContext('2d'), {{
            type: 'line',
            data: {{
                labels: {labels_json},
                datasets: [
                    {{ label: 'ストレス', data: {stress_json}, borderColor: '#ff6b6b', backgroundColor: 'rgba(255,107,107,0.1)', tension: 0.4, fill: true }},
                    {{ label: 'モチベ', data: {motivation_json}, borderColor: '#a8edea', backgroundColor: 'rgba(168,237,234,0.1)', tension: 0.4, fill: true }}
                ]
            }},
            options: {{
                responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ labels: {{ color: '#eee' }} }} }},
                scales: {{
                    x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }}, min: 0, max: 10 }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    return HTMLResponse(html)


# ===========================================================================
# 静的ファイル配信 & ヘルスチェック
# ===========================================================================
@app.get("/health")
def health_check():
    """Render のヘルスチェック用エンドポイント。"""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_index():
    """ルートに index.html を返す。"""
    index_path = BASE_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

