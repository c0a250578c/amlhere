"""
Amlhere（アムリエ）- 感情分析 & ダッシュボード API
感情ログの見える化・集計・AIフィードバック
"""

import json
import logging
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from html import escape as html_escape
from typing import Generator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from google import genai

load_dotenv()

# ---------------------------------------------------------------------------
# ログ設定
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("amlhere.analytics")

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
DB_PATH = os.environ.get("AMLHERE_DB_PATH", "chappy.db")
GEMINI_MODEL = os.environ.get("AMLHERE_MODEL", "gemini-2.5-flash-preview-04-17")
ALLOWED_ORIGINS = os.environ.get(
    "AMLHERE_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
MAX_DAYS = 365

# ---------------------------------------------------------------------------
# FastAPI アプリ
# ---------------------------------------------------------------------------
app = FastAPI(title="Amlhere Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Gemini クライアント
# ---------------------------------------------------------------------------
_gemini_api_key = os.environ.get("GEMINI_API_KEY")
if not _gemini_api_key:
    logger.warning("GEMINI_API_KEY が設定されていません")
gemini_client = genai.Client(api_key=_gemini_api_key)


# ---------------------------------------------------------------------------
# バリデーション
# ---------------------------------------------------------------------------
def _validate_user_id(user_id: str) -> str:
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
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 感情スコア変換
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# DB からの感情ログ取得・集計
# ---------------------------------------------------------------------------
def get_emotion_logs(user_id: str, days: int = 7) -> list[dict]:
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


def aggregate_by_day(logs: list[dict]) -> list[dict]:
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


# ---------------------------------------------------------------------------
# AI フィードバック生成
# ---------------------------------------------------------------------------
def _generate_feedback(daily: list[dict], days: int) -> str:
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
        return response.text.strip()
    except Exception:
        logger.exception("フィードバック生成に失敗しました")
        return "フィードバックを生成できませんでした。しばらくしてからお試しください。"


# ---------------------------------------------------------------------------
# GET /analytics/{user_id}
# ---------------------------------------------------------------------------
@app.get("/analytics/{user_id}")
def get_analytics(
    user_id: str,
    days: int = Query(default=7, ge=1, le=MAX_DAYS),
):
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


# ---------------------------------------------------------------------------
# GET /feedback/{user_id}
# ---------------------------------------------------------------------------
@app.get("/feedback/{user_id}")
def get_feedback(
    user_id: str,
    days: int = Query(default=7, ge=1, le=MAX_DAYS),
):
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


# ---------------------------------------------------------------------------
# GET /dashboard/{user_id}  — ダッシュボード HTML
# ---------------------------------------------------------------------------
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

    # グラフ用データ
    labels = [d["date"] for d in daily]
    stress_data = [d["avg_stress"] for d in daily]
    motivation_data = [d["avg_motivation"] for d in daily]

    avg_stress = round(sum(stress_data) / len(stress_data), 1)
    avg_motivation = round(sum(motivation_data) / len(motivation_data), 1)

    # AI フィードバック（HTML エスケープして安全に埋め込む）
    feedback = html_escape(_generate_feedback(daily, days))

    # JSON はテンプレートに安全に埋め込む
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
            min-height: 100vh;
            color: #eee;
            padding: 30px;
        }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2rem;
            margin-bottom: 8px;
            background: linear-gradient(90deg, #a8edea, #fed6e3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
            font-size: 0.9rem;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            font-size: 1rem;
            color: #a8edea;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }}
        .feedback {{
            font-size: 1.1rem;
            line-height: 1.8;
            color: #eee;
            border-left: 3px solid #a8edea;
            padding-left: 16px;
        }}
        .chart-container {{ position: relative; height: 250px; }}
        .stats {{ display: flex; gap: 16px; margin-bottom: 20px; }}
        .stat {{
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2rem; font-weight: bold; color: #a8edea; }}
        .stat-label {{ font-size: 0.8rem; color: #aaa; margin-top: 4px; }}
        @media (max-width: 600px) {{
            body {{ padding: 16px; }}
            .stats {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Amlhere</h1>
        <p class="subtitle">過去{days}日間の感情レポート / {safe_user_id}</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{avg_stress}</div>
                <div class="stat-label">平均ストレス</div>
            </div>
            <div class="stat">
                <div class="stat-value">{avg_motivation}</div>
                <div class="stat-label">平均モチベ</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(logs)}</div>
                <div class="stat-label">総記録数</div>
            </div>
        </div>

        <div class="card">
            <h2>Amlhereからのフィードバック</h2>
            <p class="feedback">{feedback}</p>
        </div>

        <div class="card">
            <h2>感情の推移</h2>
            <div class="chart-container">
                <canvas id="chart"></canvas>
            </div>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels_json},
                datasets: [
                    {{
                        label: 'ストレス',
                        data: {stress_json},
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255,107,107,0.1)',
                        tension: 0.4,
                        fill: true
                    }},
                    {{
                        label: 'モチベ',
                        data: {motivation_json},
                        borderColor: '#a8edea',
                        backgroundColor: 'rgba(168,237,234,0.1)',
                        tension: 0.4,
                        fill: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ labels: {{ color: '#eee' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#aaa' }}, grid: {{ color: 'rgba(255,255,255,0.05)' }} }},
                    y: {{
                        ticks: {{ color: '#aaa' }},
                        grid: {{ color: 'rgba(255,255,255,0.05)' }},
                        min: 0, max: 10
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""
    return HTMLResponse(html)
