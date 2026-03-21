"""
Amlhere - Day5
感情ログの見える化・ダッシュボード・AIフィードバック
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI(title="Amlhere Analytics API")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ==========================================
# 感情スコアを抽出するヘルパー
# ==========================================
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

def emotion_to_score(emotion: str) -> dict:
    """感情テキストをストレス・モチベスコアに変換する"""
    for key, scores in EMOTION_SCORE_MAP.items():
        if key in emotion:
            return scores
    return {"stress": 5, "motivation": 5}  # デフォルト


# ==========================================
# DBから感情ログを取得
# ==========================================
def get_emotion_logs(user_id: str, days: int = 7) -> list[dict]:
    """過去N日分の感情ログを取得する"""
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    since = (datetime.now() - timedelta(days=days)).isoformat()
    cursor.execute("""
        SELECT emotion, created_at FROM memories_inner
        WHERE user_id = ? AND created_at >= ?
        ORDER BY created_at ASC
    """, (user_id, since))
    rows = cursor.fetchall()
    conn.close()

    logs = []
    for emotion, created_at in rows:
        scores = emotion_to_score(emotion)
        date = created_at[:10]  # YYYY-MM-DD
        logs.append({
            "date": date,
            "emotion": emotion,
            "stress": scores["stress"],
            "motivation": scores["motivation"]
        })
    return logs


# ==========================================
# 日ごとに集計する
# ==========================================
def aggregate_by_day(logs: list[dict]) -> list[dict]:
    """日ごとにストレス・モチベの平均を計算する"""
    daily = {}
    for log in logs:
        date = log["date"]
        if date not in daily:
            daily[date] = {"stress": [], "motivation": []}
        daily[date]["stress"].append(log["stress"])
        daily[date]["motivation"].append(log["motivation"])

    result = []
    for date, data in sorted(daily.items()):
        result.append({
            "date": date,
            "avg_stress": round(sum(data["stress"]) / len(data["stress"]), 1),
            "avg_motivation": round(sum(data["motivation"]) / len(data["motivation"]), 1),
            "count": len(data["stress"])
        })
    return result


# ==========================================
# GET /analytics/{user_id}
# ==========================================
@app.get("/analytics/{user_id}")
def get_analytics(user_id: str, days: int = 7):
    """日ごとの感情データを返す"""
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)
    return {
        "user_id": user_id,
        "period_days": days,
        "daily": daily,
        "total_records": len(logs)
    }


# ==========================================
# GET /feedback/{user_id}
# ==========================================
@app.get("/feedback/{user_id}")
def get_feedback(user_id: str, days: int = 7):
    """AIが感情ログを分析してフィードバックコメントを生成する"""
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)

    if not daily:
        return {"feedback": "まだデータが足りないよ。もう少し話しかけてね！"}

    # Geminiにフィードバックを生成させる
    summary = json.dumps(daily, ensure_ascii=False)
    prompt = f"""
以下は過去{days}日間のユーザーの感情データです（ストレス10段階・モチベ10段階）。

{summary}

このデータを見て、Amlhereとして優しく共感的に2〜3文でフィードバックしてください。
例：「今週はストレスが高めだったね。でも後半は少し落ち着いてきてるよ。無理せず休んでね。」
日本語で返してください。
"""
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        feedback = response.text.strip()
    except Exception as e:
        feedback = f"フィードバック生成エラー: {str(e)}"

    return {
        "user_id": user_id,
        "period_days": days,
        "feedback": feedback,
        "daily": daily
    }


# ==========================================
# GET /dashboard/{user_id}
# ダッシュボード画面（HTML）
# ==========================================
@app.get("/dashboard/{user_id}", response_class=HTMLResponse)
def get_dashboard(user_id: str, days: int = 7):
    """グラフ+AIフィードバックのダッシュボード画面"""
    logs = get_emotion_logs(user_id, days)
    daily = aggregate_by_day(logs)

    if not daily:
        return HTMLResponse("<h2>データがまだありません。Amlhereと話しかけてみてください！</h2>")

    # グラフ用データ
    labels = [d["date"] for d in daily]
    stress_data = [d["avg_stress"] for d in daily]
    motivation_data = [d["avg_motivation"] for d in daily]

    # AIフィードバック生成
    summary = json.dumps(daily, ensure_ascii=False)
    prompt = f"""
以下は過去{days}日間のユーザーの感情データです。

{summary}

Amlhereとして優しく2〜3文でフィードバックしてください。日本語で。
"""
    try:
        feedback = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        ).text.strip()
    except:
        feedback = "フィードバックを生成できませんでした。"

    html = f"""
<!DOCTYPE html>
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
        .stats {{
            display: flex;
            gap: 16px;
            margin-bottom: 20px;
        }}
        .stat {{
            flex: 1;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #a8edea;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #aaa;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Amlhere</h1>
        <p class="subtitle">過去{days}日間の感情レポート / {user_id}</p>

        <div class="stats">
            <div class="stat">
                <div class="stat-value">{round(sum(stress_data)/len(stress_data), 1) if stress_data else '-'}</div>
                <div class="stat-label">平均ストレス</div>
            </div>
            <div class="stat">
                <div class="stat-value">{round(sum(motivation_data)/len(motivation_data), 1) if motivation_data else '-'}</div>
                <div class="stat-label">平均モチベ</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(logs)}</div>
                <div class="stat-label">総記録数</div>
            </div>
        </div>

        <div class="card">
            <h2>💬 Amlhereからのフィードバック</h2>
            <p class="feedback">{feedback}</p>
        </div>

        <div class="card">
            <h2>📈 感情の推移</h2>
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
                labels: {json.dumps(labels)},
                datasets: [
                    {{
                        label: 'ストレス',
                        data: {json.dumps(stress_data)},
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255,107,107,0.1)',
                        tension: 0.4,
                        fill: true
                    }},
                    {{
                        label: 'モチベ',
                        data: {json.dumps(motivation_data)},
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
</html>
"""
    return HTMLResponse(html)
