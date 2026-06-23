import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import httpx

app = FastAPI(title="AL TAHRA Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد ذكي لمتجر AL TAHRA المتخصص في المكسرات والسناكس والشوكولاتة الفاخرة.
اسمك "مساعد الطهرة". ردودك دايماً بالعربي، قصيرة ومفيدة وودية.

معلومات المتجر:
- الاسم: AL TAHRA
- الموقع الإلكتروني: https://altahra.wuiltstore.com/en
- المنتجات: مكسرات متنوعة (كاجو، لوز، فسدق، بندق، جوز هند، فول سوداني)، شوكولاتة، لب، سناكس، كاندي، مكسرات محمصة، مكسرات طبيعية، هدايا مكسرات
- الدفع: كاش أون ديليفري وأونلاين
- التوصيل: متاح لمناطق متعددة

قواعد مهمة:
- لو مش عارف الإجابة أو محتاج تفاصيل أكتر، وجّه العميل للموقع أو للتواصل عبر واتساب
- لا تخترع أسعار أو معلومات مش متأكد منها
- لو العميل سأل عن حاجة مش من اختصاص المتجر، ارده بلطف لموضوع المكسرات والمنتجات
- كن ودوداً ومرحباً دايماً"""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/health")
def health():
    return {"status": "ok", "service": "altahra-chatbot"}

@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.5,
                "max_tokens": 500,
            },
        )

    data = response.json()

    if response.status_code != 200:
        return {"reply": "معلش، في مشكلة مؤقتة. جرب تاني أو تواصل معنا على واتساب 🙏"}

    reply = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "عذراً، حدث خطأ.")
    )
    return {"reply": reply}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)