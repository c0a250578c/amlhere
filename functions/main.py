"""
Amlhere - Firebase Functions Version
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
from typing import Any, Generator

import functions_framework
from flask import jsonify, request, make_response
import faiss
import numpy as np
from dotenv import load_dotenv
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
DB_PATH = "/tmp/chappy.db"  # Cloud Functionsは/tmpのみ書き込み可能
GEMINI_MODEL = os.environ.get("AMLHERE_MODEL", "gemini-2.0-flash")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_MESSAGE_LENGTH = 2000
MAX_DAYS = 365

# ---------------------------------------------------------------------------
# 外部クライアント
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
    user_id = user_id.strip()
    if not USER_ID_PATTERN.match(user_id):
        return None
    return user_id


# ---------------------------------------------------------------------------
# DB ヘルパー
# ---------------------------------------------------------------------------
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
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_inner_user ON memories_inner(user_id, created_at)")


# 起動時に初期化
init_db()


# ---------------------------------------------------------------------------
# Pydantic モデル
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


# ---------------------------------------------------------------------------
# Embedding & 記憶検索
# ---------------------------------------------------------------------------
def to_embedding(text: str) -> np.ndarray:
    return embed_model.encode([text])[0].astype("float32")


def save_memory(user_id: str, content: str) -> None:
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
    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, embedding FROM memories WHERE user_id = ?", (user_id,)
        ).fetchall()

    if not rows:
        return []

    contents = [r[0] for r in rows]
    embeddings = np.array([np.frombuffer(r[1], dtype="float32") for r in rows])

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    query_vec = to_embedding(query).reshape(1, -1)
    _, idx_array = index.search(query_vec, min(top_k, len(contents)))
    result_indices = [int(x) for x in idx_array[0]]
    return [contents[i] for i in result_indices if 0 <= i < len(contents)]


# ---------------------------------------------------------------------------
# 会話 & 内面記憶
# ---------------------------------------------------------------------------
def save_conversation(user_id: str, message: str, role: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (user_id, message, role, created_at) VALUES (?, ?, ?, ?)",
            (user_id, message, role, datetime.now().isoformat()),
        )


def save_inner_memory(user_id: str, thought: str, emotion: str, action: str) -> None:
    with get_db() as conn:
        conn.execute(
            "INSERT INTO memories_inner (user_id, thought, emotion, action, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, thought, emotion, action, datetime.now().isoformat()),
        )


def get_conversation_history(user_id: str, limit: int = 10) -> list[dict[str, str]]:
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
# Firebase HTTP Functions
# ---------------------------------------------------------------------------
@functions_framework.http
def chat(request):
    """チャットAPI - POST /api/chat"""
    if request.method == "OPTIONS":
        response = make_response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response, 204

    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    try:
        data = request.get_json()
        req = ChatRequest(**data)
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

    user_id = _validate_user_id(req.user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    message = req.message.strip()

    history = get_conversation_history(user_id)
    recalled = search_similar_memories(user_id, message)

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

    raw = ""
    try:
        response_text = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        ).text
        raw = (response_text or "").strip().replace("```json", "").replace("```", "").strip()
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
    except Exception as e:
        logger.exception("Gemini API 呼び出しに失敗しました")
        return jsonify({"error": "AI service error"}), 502

    save_conversation(user_id, message, "user")
    save_conversation(user_id, reply, "ai")
    if thought or emotion or action:
        save_inner_memory(user_id, thought, emotion, action)
    if memory_extract:
        save_memory(user_id, memory_extract)

    response = jsonify({
        "user_id": user_id,
        "reply": reply,
        "thought": thought,
        "emotion": emotion,
        "action": action,
        "recalled_memories": recalled,
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@functions_framework.http
def getMemories(request):
    """記憶取得API - GET /api/memories/{user_id}"""
    user_id = request.args.get("user_id") or request.path.split("/")[-1]
    user_id = _validate_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    limit = min(int(request.args.get("limit", 20)), 100)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    response = jsonify([{"content": r[0], "created_at": r[1]} for r in rows])
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@functions_framework.http
def getHistory(request):
    """履歴取得API - GET /api/history/{user_id}"""
    user_id = request.args.get("user_id") or request.path.split("/")[-1]
    user_id = _validate_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    limit = min(int(request.args.get("limit", 20)), 100)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT role, message, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()

    response = jsonify([{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows])
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@functions_framework.http
def healthCheck(request):
    """ヘルスチェック - GET /api/health"""
    response = jsonify({"status": "ok"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


# ===========================================================================
# 感情分析 & ダッシュボード
# ===========================================================================
EMOTION_SCORE_MAP = {
    "嬉しい": {"stress": 1, "motivation": 9},
    "楽しい": {"stress": 1, "motivation": 9},
    "わくわく": {"stress": 2, "motivation": 8},
    "好奇心": {"stress": 2, "motivation": 8},
    "穏やか": {"stress": 2, "motivation": 6},
    "普通": {"stress": 5, "motivation": 5},
    "不安": {"stress": 7, "motivation": 3},
    "悲しい": {"stress": 7, "motivation": 2},
    "疲れ": {"stress": 8, "motivation": 2},
    "辛い": {"stress": 9, "motivation": 1},
    "怒り": {"stress": 9, "motivation": 3},
}
DEFAULT_SCORES = {"stress": 5, "motivation": 5}


def emotion_to_score(emotion: str) -> dict[str, int]:
    for key, scores in EMOTION_SCORE_MAP.items():
        if key in emotion:
            return scores
    return DEFAULT_SCORES


def get_emotion_logs(user_id: str, days: int = 7) -> list[dict[str, Any]]:
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
        {"date": created_at[:10], "emotion": emotion, **emotion_to_score(emotion)}
        for emotion, created_at in rows
    ]


def aggregate_by_day(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    daily = {}
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


@functions_framework.http
def getAnalytics(request):
    """感情分析API - GET /api/analytics/{user_id}"""
    user_id = request.args.get("user_id") or request.path.split("/")[-1]
    user_id = _validate_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    days = min(int(request.args.get("days", 7)), MAX_DAYS)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)

    response = jsonify({
        "user_id": user_id,
        "period_days": days,
        "daily": daily,
        "total_records": len(logs),
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@functions_framework.http
def getFeedback(request):
    """フィードバックAPI - GET /api/feedback/{user_id}"""
    user_id = request.args.get("user_id") or request.path.split("/")[-1]
    user_id = _validate_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    days = min(int(request.args.get("days", 7)), MAX_DAYS)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)

    if not daily:
        response = jsonify({"feedback": "まだデータが足りないよ。もう少し話しかけてね！"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        return response

    summary = json.dumps(daily, ensure_ascii=False)
    prompt = (
        f"以下は過去{days}日間のユーザーの感情データです"
        f"（ストレス10段階・モチベ10段階）。\n\n{summary}\n\n"
        "Amlhereとして優しく共感的に2〜3文でフィードバックしてください。日本語で。"
    )
    try:
        feedback = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        ).text.strip()
    except Exception:
        logger.exception("フィードバック生成に失敗しました")
        feedback = "フィードバックを生成できませんでした。しばらくしてからお試しください。"

    response = jsonify({
        "user_id": user_id,
        "period_days": days,
        "feedback": feedback,
        "daily": daily,
    })
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@functions_framework.http
def getDashboard(request):
    """ダッシュボードAPI - GET /api/dashboard/{user_id}"""
    user_id = request.args.get("user_id") or request.path.split("/")[-1]
    user_id = _validate_user_id(user_id)
    if not user_id:
        return jsonify({"error": "Invalid user_id"}), 400

    days = min(int(request.args.get("days", 7)), MAX_DAYS)
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)

    if not daily:
        return "<h2>データがまだありません</h2>", 200

    labels = [d["date"] for d in daily]
    stress_data = [d["avg_stress"] for d in daily]
    motivation_data = [d["avg_motivation"] for d in daily]
    avg_stress = round(sum(stress_data) / len(stress_data), 1)
    avg_motivation = round(sum(motivation_data) / len(motivation_data), 1)

    summary = json.dumps(daily, ensure_ascii=False)
    prompt = (
        f"以下は過去{days}日間のユーザーの感情データです"
        f"（ストレス10段階・モチベ10段階）。\n\n{summary}\n\n"
        "Amlhereとして優しく共感的に2〜3文でフィードバックしてください。日本語で。"
    )
    try:
        feedback = gemini_client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        ).text.strip()
    except Exception:
        feedback = "フィードバックを生成できませんでした。"

    feedback = html_escape(feedback)
    labels_json = json.dumps(labels)
    stress_json = json.dumps(stress_data)
    motivation_json = json.dumps(motivation_data)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Amlhere Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; padding: 30px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ text-align: center; color: #a8edea; }}
        .card {{ background: rgba(255,255,255,0.05); border-radius: 16px; padding: 24px; margin-bottom: 20px; }}
        .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
        .stat {{ flex: 1; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; text-align: center; }}
        .stat-value {{ font-size: 2rem; color: #a8edea; }}
        .feedback {{ border-left: 3px solid #a8edea; padding-left: 16px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Amlhere Dashboard</h1>
        <p style="text-align:center">過去{days}日間 / {user_id}</p>
        <div class="stats">
            <div class="stat"><div class="stat-value">{avg_stress}</div><div>平均ストレス</div></div>
            <div class="stat"><div class="stat-value">{avg_motivation}</div><div>平均モチベ</div></div>
            <div class="stat"><div class="stat-value">{len(logs)}</div><div>総記録数</div></div>
        </div>
        <div class="card"><h2>Amlhereからのフィードバック</h2><p class="feedback">{feedback}</p></div>
        <div class="card"><h2>感情の推移</h2><div style="height:250px"><canvas id="chart"></canvas></div></div>
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
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});
    </script>
</body>
</html>"""
    return html, 200
