import os
from dotenv import load_dotenv
load_dotenv()

import json
import sqlite3
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

app = FastAPI(title="チャッピー API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

CHAPPY_SYSTEM_PROMPT = """
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
返答の中で自分の名前（Amlhere）を自然に使ってください。例：『Amlhereはあなたの味方です』
【重要：必ず以下のJSON形式のみで返答してください】
{"reply":"ユーザーへの返答","thought":"内なる声","emotion":"感情","action":"次にしたいこと","memory_extract":"記憶すべき重要情報"}
JSONのみ返してください。説明文やコードブロック記号は含めないでください。
"""

def init_db():
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, message TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS memories_inner (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, thought TEXT NOT NULL, emotion TEXT NOT NULL, action TEXT NOT NULL, created_at TEXT NOT NULL)""")
    cursor.execute("""CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL, created_at TEXT NOT NULL)""")
    conn.commit()
    conn.close()

init_db()

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    user_id: str
    reply: str
    thought: str
    emotion: str
    action: str
    recalled_memories: list[str]

def to_embedding(text):
    return embed_model.encode([text])[0].astype("float32")

def save_memory(user_id, content):
    if not content.strip():
        return
    embedding = to_embedding(content).tobytes()
    conn = sqlite3.connect("chappy.db")
    conn.cursor().execute("INSERT INTO memories (user_id, content, embedding, created_at) VALUES (?, ?, ?, ?)", (user_id, content, embedding, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def search_similar_memories(user_id, query, top_k=3):
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding FROM memories WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []
    contents = [r[0] for r in rows]
    embeddings = np.array([np.frombuffer(r[1], dtype="float32") for r in rows])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _, indices = index.search(to_embedding(query).reshape(1, -1), min(top_k, len(contents)))
    return [contents[i] for i in indices[0] if i < len(contents)]

def save_conversation(user_id, message, role):
    conn = sqlite3.connect("chappy.db")
    conn.cursor().execute("INSERT INTO conversations (user_id, message, role, created_at) VALUES (?, ?, ?, ?)", (user_id, message, role, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def save_inner_memory(user_id, thought, emotion, action):
    conn = sqlite3.connect("chappy.db")
    conn.cursor().execute("INSERT INTO memories_inner (user_id, thought, emotion, action, created_at) VALUES (?, ?, ?, ?, ?)", (user_id, thought, emotion, action, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_conversation_history(user_id, limit=10):
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    rows.reverse()
    return [{"role": r[0], "message": r[1]} for r in rows]

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    history = get_conversation_history(req.user_id)
    recalled = search_similar_memories(req.user_id, req.message)
    prompt = CHAPPY_SYSTEM_PROMPT
    if recalled:
        prompt += "\n\n【このユーザーに関する記憶】\n" + "\n".join([f"- {m}" for m in recalled])
    if history:
        prompt += "\n\n【会話履歴】\n" + "\n".join([f"{'ユーザー' if h['role']=='user' else 'チャッピー'}：{h['message']}" for h in history])
    prompt += f"\n\nユーザー：{req.message}\nチャッピー："
    try:
        raw = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt).text.strip().replace("```json","").replace("```","").strip()
        data = json.loads(raw)
        reply = data.get("reply","...")
        thought = data.get("thought","")
        emotion = data.get("emotion","")
        action = data.get("action","")
        memory_extract = data.get("memory_extract","")
    except json.JSONDecodeError:
        reply = raw
        thought = emotion = action = memory_extract = ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini APIエラー: {str(e)}")
    save_conversation(req.user_id, req.message, "user")
    save_conversation(req.user_id, reply, "ai")
    if thought or emotion or action:
        save_inner_memory(req.user_id, thought, emotion, action)
    if memory_extract:
        save_memory(req.user_id, memory_extract)
    return ChatResponse(user_id=req.user_id, reply=reply, thought=thought, emotion=emotion, action=action, recalled_memories=recalled)

@app.get("/memories/{user_id}")
def get_memories(user_id: str, limit: int = 20):
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    cursor.execute("SELECT content, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"content": r[0], "created_at": r[1]} for r in rows]

@app.get("/history/{user_id}")
def get_history(user_id: str, limit: int = 20):
    conn = sqlite3.connect("chappy.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, message, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows]

