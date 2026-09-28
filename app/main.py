from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.config import VLLM_MODEL
from app.graph import agent
from app.llm import chat_image, client

app = FastAPI(title="Document Intelligence Assistant")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    agent_path: list[str]


@app.get("/health")
def health():
    try:
        models = [m.id for m in client.models.list().data]
        return {"status": "ok", "vllm_models": models}
    except Exception as e:
        return {"status": "degraded", "error": str(e), "expected_model": VLLM_MODEL}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(400, "question must not be empty")
    try:
        out = agent.invoke({"question": req.question})
    except Exception as e:
        raise HTTPException(503, f"Backend error: {e}")
    return AskResponse(answer=out["answer"], sources=out.get("sources", []), agent_path=out["path"])


@app.post("/ask-image")
async def ask_image(question: str = Form(...), image: UploadFile = File(...)):
    data = await image.read()
    try:
        ans = chat_image(data, question, mime=image.content_type or "image/png")
    except Exception as e:
        raise HTTPException(503, f"Backend error: {e}")
    return {"answer": ans, "agent_path": ["visual_qa"]}
