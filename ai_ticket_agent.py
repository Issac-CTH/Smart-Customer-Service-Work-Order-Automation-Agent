# ai_ticket_agent.py
import os
import re
import uuid
import time
import json
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
import numpy as np

# Optional: use OpenAI if OPENAI_API_KEY set
try:
    import openai
except Exception:
    openai = None

# Embedding model (sentence-transformers)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Config
DB_PATH = os.environ.get("AGENT_DB_PATH", "sqlite:///./agent.db")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

# SQLAlchemy setup
Base = declarative_base()
engine = sa.create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class Ticket(Base):
    __tablename__ = "tickets"
    id = sa.Column(sa.String, primary_key=True, index=True)
    user_id = sa.Column(sa.String, index=True)
    status = sa.Column(sa.String, default="open")  # open, pending_human, closed
    intent = sa.Column(sa.String, nullable=True)
    slots = sa.Column(sa.Text, nullable=True)  # json
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)
    updated_at = sa.Column(sa.DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = sa.Column(sa.String, primary_key=True, index=True)
    ticket_id = sa.Column(sa.String, index=True)
    role = sa.Column(sa.String)  # user, agent, system
    text = sa.Column(sa.Text)
    meta = sa.Column(sa.Text, nullable=True)  # json
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)

class KBArticle(Base):
    __tablename__ = "kb"
    id = sa.Column(sa.String, primary_key=True, index=True)
    title = sa.Column(sa.String)
    content = sa.Column(sa.Text)
    embedding = sa.Column(sa.Text)  # json list

Base.metadata.create_all(bind=engine)

# Populate sample KB if empty
def seed_kb():
    s = SessionLocal()
    exist = s.query(KBArticle).first()
    if exist:
        s.close()
        return
    kb_samples = [
        {"id": "kb_refund_policy", "title": "退款政策", "content": "订单在7天内可申请退款，退款会退回原支付渠道，通常3-7个工作日到账。需要订单号和退款原因。"},
        {"id": "kb_invoice_process", "title": "开票流程", "content": "如需发票，请提供抬头、税号、订单号及金额。电子发票一般1-3个工作日开出。"},
        {"id": "kb_password_reset", "title": "密码重置", "content": "用户可以通过忘记密码功能用注册邮箱或手机号重置密码；若无法接收验证码，请联系客服协助。"},
        {"id": "kb_cancel_order", "title": "取消订单", "content": "未发货订单支持取消；发货后需走退货流程。取消请提供订单号。"},
    ]
    texts = [k["content"] for k in kb_samples]
    embeds = EMBED_MODEL.encode(texts, convert_to_numpy=True)
    for k, e in zip(kb_samples, embeds):
        kb = KBArticle(id=k["id"], title=k["title"], content=k["content"], embedding=json.dumps(e.tolist()))
        s.add(kb)
    s.commit()
    s.close()

seed_kb()

# Pydantic models
class IncomingMessage(BaseModel):
    user_id: str
    text: str
    channel: Optional[str] = "web"
    ticket_id: Optional[str] = None

# Simple utility functions
def now_ts():
    return datetime.utcnow()

def gen_id(prefix="id"):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

def load_ticket(db, ticket_id):
    return db.query(Ticket).filter(Ticket.id == ticket_id).first()

def save_message(db, ticket_id, role, text, meta=None):
    m = Message(id=gen_id("msg"), ticket_id=ticket_id, role=role, text=text, meta=json.dumps(meta) if meta else None, created_at=now_ts())
    db.add(m)
    db.commit()
    return m

# KB retrieval
def retrieve_kb(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    s = SessionLocal()
    articles = s.query(KBArticle).all()
    s.close()
    if not articles:
        return []
    texts = [a.content for a in articles]
    embeddings = [np.array(json.loads(a.embedding)) for a in articles]
    q_emb = EMBED_MODEL.encode([query], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        a = articles[i]
        results.append({"id": a.id, "title": a.title, "content": a.content, "score": float(sims[i])})
    return results

# NLU: intent + slot extraction (uses OpenAI if available, else rule-based fallback)
AUTO_INTENTS = {"refund", "invoice_request", "password_reset", "cancel_order", "information"}

def nlu_parse(text: str) -> Dict[str, Any]:
    """
    Return dict: {intent: str, intent_confidence: float, slots: {slot: value}, missing_slots: []}
    """
    # Try using OpenAI if available
    if openai and OPENAI_API_KEY:
        prompt = f"""
        对用户一句话进行意图识别与槽位抽取。输出为 JSON: {{ "intent": "...", "confidence": 0.0~1.0, "slots":{{...}} }}。
        槽位包括：order_id, amount, reason, invoice_title, tax_id, email, phone, password_reset_method (email/sms)。
        用户输入: \"\"\"{text}\"\"\"
        只输出严格的 JSON。
        """
        try:
            resp = openai.ChatCompletion.create(model="gpt-4o-mini" if False else "gpt-3.5-turbo", messages=[{"role":"system","content":"你是意图与槽位解析器。"},{"role":"user","content":prompt}], temperature=0)
            content = resp["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            parsed.setdefault("intent_confidence", parsed.get("confidence", 0.7))
            return {"intent": parsed.get("intent","information"), "intent_confidence": float(parsed.get("intent_confidence",0.7)), "slots": parsed.get("slots",{})}
        except Exception as e:
            # fallback to rule-based
            pass

    # Rule-based fallback
    text_low = text.lower()
    intent = "information"
    if any(w in text_low for w in ["退款","退钱","退单","return","refund"]):
        intent = "refund"
    elif any(w in text_low for w in ["发票","开票","票据","invoice"]):
        intent = "invoice_request"
    elif any(w in text_low for w in ["忘记密码","重置密码","密码重置","reset password"]):
        intent = "password_reset"
    elif any(w in text_low for w in ["取消订单","取消","cancel order"]):
        intent = "cancel_order"
    # slots via regex
    slots = {}
    m = re.search(r"订单\s*#?\s*([A-Za-z0-9\-]+)", text, re.I)
    if not m:
        m = re.search(r"订单\s*号[:：]?\s*([A-Za-z0-9\-]+)", text, re.I)
    if m:
        slots["order_id"] = m.group(1)
    m2 = re.search(r"(\d+\.?\d*)\s*(元|rmb|cny|￥)", text_low)
    if m2:
        slots["amount"] = m2.group(1)
    # email
    m3 = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
    if m3:
        slots["email"] = m3.group(1)
    # phone
    m4 = re.search(r"((?:\+?\d{2,4}[- ]?)?\d{6,12})", text)
    if m4 and len(m4.group(1)) >= 6:
        slots["phone"] = m4.group(1)
    # invoice details heuristic
    if intent == "invoice_request":
        m5 = re.search(r"抬头[:：]?\s*([^,，。]+)", text)
        if m5:
            slots["invoice_title"] = m5.group(1).strip()
        m6 = re.search(r"税号[:：]?\s*([A-Za-z0-9]+)", text)
        if m6:
            slots["tax_id"] = m6.group(1).strip()
    return {"intent": intent, "intent_confidence": 0.8, "slots": slots}

# Determine required slots per intent
REQUIRED_SLOTS = {
    "refund": ["order_id", "amount", "reason"],
    "invoice_request": ["order_id", "invoice_title", "tax_id", "amount"],
    "password_reset": ["email_or_phone",],  # accept email or phone
    "cancel_order": ["order_id"],
    "information": []
}

def missing_slots_for(intent: str, slots: Dict[str,Any]) -> List[str]:
    req = REQUIRED_SLOTS.get(intent, [])
    missing = []
    for r in req:
        if r == "email_or_phone":
            if not (slots.get("email") or slots.get("phone")):
                missing.append(r)
        else:
            if not slots.get(r):
                missing.append(r)
    return missing

# Draft reply generation (uses OpenAI if available, else template)
def generate_draft_reply(user_text: str, intent: str, slots: Dict[str,Any], kb_context: List[Dict[str,Any]], extra: Optional[Dict]=None) -> str:
    kb_snippets = "\n".join([f"- {k['title']}: {k['content']}" for k in kb_context])
    if openai and OPENAI_API_KEY:
        prompt = f"""
        你是客服助手。根据用户输入: \"\"\"{user_text}\"\"\"，意图: {intent}，已抽取槽位: {json.dumps(slots, ensure_ascii=False)}。
        可参考的知识库片段:
        {kb_snippets}
        请输出一段友好、简洁、专业的回复草稿；若需要用户提供更多信息，明确列出需要的缺失字段。不要包含多余说明。
        """
        try:
            resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"system","content":"你是客服助理。"},{"role":"user","content":prompt}], temperature=0.2)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
    # Template fallback
    if intent == "refund":
        order = slots.get("order_id","<未提供订单号>")
        amount = slots.get("amount","<未提供金额>")
        return f"您好，已收到您申请退款的请求。订单号 {order}，退款金额 {amount}。请提供退款原因（如商品损坏/不满意等），我们会在审核后3-7个工作日将款项退回原支付渠道。"
    if intent == "invoice_request":
        return f"您好，关于开票我们需要：订单号、发票抬头、税号以及金额。请提供相关信息，我们将在1-3个工作日内为您开具电子发票。"
    if intent == "password_reset":
        return "您好，您可通过“忘记密码”功能重置。如果无法收到验证码，请告知是注册邮箱还是手机号，我们可以协助人工重置。"
    return "您好，请描述您的问题，我会尽快为您处理。"

# Router: decide auto or manual
def decide_auto_or_manual(intent: str, intent_conf: float, slots: Dict[str,Any]) -> Dict[str,Any]:
    """Return dict: {mode: 'auto'|'manual', reason: str}"""
    # Basic policy: if intent recognized as one of AUTO_INTENTS and required slots are present, allow auto
    if intent in AUTO_INTENTS:
        missing = missing_slots_for(intent, slots)
        if missing:
            return {"mode":"manual", "reason": f"缺失槽位: {missing}"}
        # For safety, require confidence threshold
        if intent_conf < 0.6:
            return {"mode":"manual", "reason":"意图置信度低"}
        # else auto
        return {"mode":"auto", "reason":"满足自动化条件"}
    # default route to manual for unknown intents
    return {"mode":"manual", "reason":"非自动化意图"}

# Automation executors (simulated)
def exec_refund(slots: Dict[str,Any]) -> Dict[str,Any]:
    # Simulate calling payment gateway
    time.sleep(0.5)
    # generate a mock refund id
    rid = gen_id("refund")
    return {"status":"success","refund_id":rid,"message":"退款已受理，预计3-7个工作日到账。"}

def exec_invoice(slots: Dict[str,Any]) -> Dict[str,Any]:
    time.sleep(0.5)
    inv_id = gen_id("inv")
    return {"status":"success","invoice_id":inv_id,"message":"电子发票已申请，1-3个工作日内下发到您邮箱。"}

def exec_password_reset(slots: Dict[str,Any]) -> Dict[str,Any]:
    time.sleep(0.2)
    return {"status":"success","message":"密码重置指令已发送（模拟）。请检查您的邮箱或短信。"}

def perform_action(intent: str, slots: Dict[str,Any]) -> Dict[str,Any]:
    if intent == "refund":
        return exec_refund(slots)
    if intent == "invoice_request":
        return exec_invoice(slots)
    if intent == "password_reset":
        return exec_password_reset(slots)
    if intent == "cancel_order":
        return {"status":"success","message":"订单取消已提交（模拟）。"}
    return {"status":"failed","message":"不支持的自动化操作。"}

# Main API server
app = FastAPI(title="AI Ticket Agent Demo")

@app.post("/api/message")
async def handle_message(msg: IncomingMessage):
    db = SessionLocal()
    # find or create ticket
    ticket = None
    if msg.ticket_id:
        ticket = load_ticket(db, msg.ticket_id)
    if not ticket:
        ticket = Ticket(id=gen_id("tkt"), user_id=msg.user_id, status="open", created_at=now_ts(), updated_at=now_ts())
        db.add(ticket)
        db.commit()
    # save user message
    save_message(db, ticket.id, "user", msg.text, meta={"channel": msg.channel})
    # Context: load previous messages optionally (not large here)
    # NLU
    nlu = nlu_parse(msg.text)
    intent = nlu["intent"]
    intent_conf = float(nlu.get("intent_confidence",0.8))
    parsed_slots = nlu.get("slots",{})
    # merge into ticket slots
    ticket_slots = {}
    if ticket.slots:
        try:
            ticket_slots = json.loads(ticket.slots)
        except:
            ticket_slots = {}
    # merge, prefer new values
    ticket_slots.update(parsed_slots)
    # compute missing
    missing = missing_slots_for(intent, ticket_slots)
    # retrieve KB
    kb_ctx = retrieve_kb(msg.text, top_k=3)
    # if missing slots -> ask for missing (multi-turn chain)
    if missing:
        # update ticket
        ticket.intent = intent
        ticket.slots = json.dumps(ticket_slots, ensure_ascii=False)
        ticket.updated_at = now_ts()
        db.add(ticket); db.commit()
        # Build prompt asking for missing fields
        req_items = ", ".join(missing)
        draft = generate_draft_reply(msg.text, intent, ticket_slots, kb_ctx)
        follow_up = f"为继续处理，请提供以下信息：{req_items}。{draft}"
        save_message(db, ticket.id, "agent", follow_up)
        db.close()
        return {"ticket_id": ticket.id, "mode": "need_slots", "missing": missing, "reply": follow_up}
    # Decide auto or manual
    decision = decide_auto_or_manual(intent, intent_conf, ticket_slots)
    if decision["mode"] == "auto":
        # perform action
        result = perform_action(intent, ticket_slots)
        # generate final reply including action result
        draft = generate_draft_reply(msg.text, intent, ticket_slots, kb_ctx)
        final_reply = f"{draft}\n\n操作结果：{result.get('message')}. 参考编号: {result.get(list(result.keys())[0]) if result.get('status')=='success' else ''}"
        # save agent message
        save_message(db, ticket.id, "agent", final_reply, meta={"auto":True,"action_result":result})
        # close ticket
        ticket.status = "closed"
        ticket.intent = intent
        ticket.slots = json.dumps(ticket_slots, ensure_ascii=False)
        ticket.updated_at = now_ts()
        db.add(ticket); db.commit()
        db.close()
        return {"ticket_id": ticket.id, "mode":"auto", "action_result": result, "reply": final_reply}
    else:
        # manual routing: prepare diagnostic and draft
        draft = generate_draft_reply(msg.text, intent, ticket_slots, kb_ctx)
        diagnostic = {
            "intent": intent,
            "intent_confidence": intent_conf,
            "slots": ticket_slots,
            "kb_references": kb_ctx,
            "reason": decision.get("reason")
        }
        ticket.status = "pending_human"
        ticket.intent = intent
        ticket.slots = json.dumps(ticket_slots, ensure_ascii=False)
        ticket.updated_at = now_ts()
        db.add(ticket); db.commit()
        save_message(db, ticket.id, "agent", draft, meta={"routed_to":"human","diagnostic":diagnostic})
        db.close()
        return {"ticket_id": ticket.id, "mode":"manual", "diagnostic": diagnostic, "suggested_reply": draft}

@app.get("/api/ticket/{ticket_id}")
def get_ticket(ticket_id: str):
    db = SessionLocal()
    ticket = load_ticket(db, ticket_id)
    if not ticket:
        db.close()
        raise HTTPException(status_code=404, detail="ticket not found")
    msgs = db.query(Message).filter(Message.ticket_id==ticket_id).order_by(Message.created_at.asc()).all()
    out_msgs = [{"role":m.role,"text":m.text,"meta": json.loads(m.meta) if m.meta else None,"created_at":m.created_at.isoformat()} for m in msgs]
    db.close()
    return {"ticket": {"id":ticket.id,"user_id":ticket.user_id,"status":ticket.status,"intent":ticket.intent,"slots": json.loads(ticket.slots) if ticket.slots else {},"created_at":ticket.created_at.isoformat(),"updated_at":ticket.updated_at.isoformat()}, "messages": out_msgs}

@app.post("/api/human_reply/{ticket_id}")
def human_reply(ticket_id: str, body: Dict[str,Any]):
    """
    Endpoint for human agent to perform reply or take action.
    body: {"agent_id": "...", "text": "...", "action": optional dict}
    """
    db = SessionLocal()
    ticket = load_ticket(db, ticket_id)
    if not ticket:
        db.close()
        raise HTTPException(status_code=404, detail="ticket not found")
    text = body.get("text","")
    # record human message
    save_message(db, ticket.id, "agent", text, meta={"human_agent": body.get("agent_id")})
    # If human performed an action, accept simulated result
    action = body.get("action")
    if action:
        # action example: {"type":"refund","result":{...}}
        save_message(db, ticket.id, "system", f"human_action:{json.dumps(action)}")
        ticket.status = "closed"
    else:
        ticket.status = "pending_human"  # still pending until closed
    ticket.updated_at = now_ts()
    db.add(ticket); db.commit()
    db.close()
    return {"status":"ok", "ticket_id": ticket_id}

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Ticket Agent on http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)