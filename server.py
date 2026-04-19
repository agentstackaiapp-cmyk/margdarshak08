from fastapi import FastAPI, APIRouter, HTTPException, Header, Response, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json as _json
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta
import httpx
from emergentintegrations.llm.chat import LlmChat, UserMessage
from youtube_video import get_embed_url

# ── Shared DB singleton (must be initialised before any service imports) ──
import db as _db_module

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
db = _db_module.init(mongo_url, os.environ['DB_NAME'])

# ── Service / router imports (after db.init) ──────────────────────────────
from services.prompt_builder import build_system_prompt
from services.user_preferences_service import get_preferences
from services.rag_service import retrieve_relevant_chunks, get_database_info, warm_cache
from services.guardrails import check_input, check_output
from routers.preferences_router import router as preferences_router

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# LLM Configuration - using Emergent Universal Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
LLM_MODEL = os.environ.get('LLM_MODEL', 'gpt-4o')

# Keep last N turns
MAX_HISTORY_TURNS = int(os.environ.get('MAX_HISTORY_TURNS', '6'))
# Max tokens the LLM generates per response
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', '2000'))

# ============= MODELS =============
class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserSession(BaseModel):
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Message(BaseModel):
    message_id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Conversation(BaseModel):
    conversation_id: str
    user_id: str
    title: str
    category: Optional[str] = None  # stress, relationships, career, ethics, spirituality
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AskQuestionRequest(BaseModel):
    question: str
    category: Optional[str] = None
    conversation_id: Optional[str] = None
    # VR-RAG: when set, the backend retrieves relevant passages from the
    # corresponding scripture PDF(s) and injects them as grounded context.
    # Valid values: "gita" | "vedas" | "upanishads" | "mahabharata" |
    #               "puran" | "all" | None (disables RAG)
    selected_database: Optional[str] = None

class SessionDataRequest(BaseModel):
    session_id: str

# ============= AUTH HELPERS =============
async def get_current_user(request: Request, authorization: Optional[str] = Header(None)) -> User:
    """Authenticate user from session_token cookie or Authorization header"""
    session_token = None
    
    # Try cookie first
    session_token = request.cookies.get("session_token")
    
    # Fallback to Authorization header
    if not session_token and authorization:
        if authorization.startswith("Bearer "):
            session_token = authorization.replace("Bearer ", "")
    
    if not session_token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Find session
    session_doc = await db.user_sessions.find_one(
        {"session_token": session_token},
        {"_id": 0}
    )
    
    if not session_doc:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check expiry
    expires_at = session_doc["expires_at"]
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    # Get user
    user_doc = await db.users.find_one(
        {"user_id": session_doc["user_id"]},
        {"_id": 0}
    )
    
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")
    
    return User(**user_doc)

# ============= AUTH ROUTES =============
class DevLoginRequest(BaseModel):
    name: str
    email: str

@api_router.post("/auth/dev-login")
async def dev_login(request: DevLoginRequest, response: Response):
    """Local dev login — creates a session directly without OAuth"""
    existing_user = await db.users.find_one({"email": request.email}, {"_id": 0})
    if existing_user:
        user_id = existing_user["user_id"]
        await db.users.update_one({"user_id": user_id}, {"$set": {"name": request.name}})
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one(User(user_id=user_id, email=request.email, name=request.name).model_dump())

    session_token = f"dev_session_{uuid.uuid4().hex}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one(UserSession(user_id=user_id, session_token=session_token, expires_at=expires_at).model_dump())

    response.set_cookie(key="session_token", value=session_token, httponly=True, secure=False, samesite="lax", path="/", max_age=7*24*60*60)
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return {**user_doc, "session_token": session_token}

@api_router.get("/auth/google/login")
async def google_login(mobile_redirect: str = None):
    """Redirect user to Google OAuth consent screen"""
    import base64, json as _json2
    from urllib.parse import urlencode
    from fastapi.responses import RedirectResponse
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
    GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")
    extra = {}
    if mobile_redirect:
        state = base64.urlsafe_b64encode(_json2.dumps({"mobile_redirect": mobile_redirect}).encode()).decode()
        extra["state"] = state
    params = urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        **extra,
    })
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


async def _upsert_user_and_session(email: str, name: str, picture: str, response: Response):
    """Create or update user, create session, set cookie, return user+token."""
    existing_user = await db.users.find_one({"email": email}, {"_id": 0})
    if existing_user:
        user_id = existing_user["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "picture": picture}}
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one(
            User(user_id=user_id, email=email, name=name, picture=picture).model_dump()
        )

    session_token = f"session_{uuid.uuid4().hex}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one(
        UserSession(user_id=user_id, session_token=session_token, expires_at=expires_at).model_dump()
    )
    response.set_cookie(
        key="session_token", value=session_token,
        httponly=True, secure=False, samesite="lax", path="/", max_age=7*24*60*60
    )
    user_doc = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    return user_doc, session_token


@api_router.get("/auth/google/callback")
async def google_callback(code: str, response: Response, state: str = ""):
    """Handle Google OAuth callback, create session, redirect to frontend."""
    GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
    GOOGLE_REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI")
    FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8084")

    try:
        async with httpx.AsyncClient() as client:
            # Exchange code for tokens
            token_resp = await client.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "code": code,
                    "client_id": GOOGLE_CLIENT_ID,
                    "client_secret": GOOGLE_CLIENT_SECRET,
                    "redirect_uri": GOOGLE_REDIRECT_URI,
                    "grant_type": "authorization_code",
                },
                timeout=10.0,
            )
            if token_resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Google token exchange failed")

            tokens = token_resp.json()
            access_token = tokens.get("access_token")

            # Fetch user info
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=10.0,
            )
            if userinfo_resp.status_code != 200:
                raise HTTPException(status_code=401, detail="Failed to fetch Google user info")

            google_user = userinfo_resp.json()

        user_doc, session_token = await _upsert_user_and_session(
            email=google_user["email"],
            name=google_user.get("name", ""),
            picture=google_user.get("picture", ""),
            response=response,
        )

        from fastapi.responses import RedirectResponse
        import base64, json as _json3
        mobile_redirect = None
        if state:
            try:
                state_data = _json3.loads(base64.urlsafe_b64decode(state.encode()).decode())
                mobile_redirect = state_data.get("mobile_redirect")
            except Exception:
                pass
        final_url = f"{mobile_redirect}#session_token={session_token}" if mobile_redirect else f"{FRONTEND_URL}/auth-callback#session_token={session_token}"
        redirect = RedirectResponse(final_url)
        redirect.set_cookie(
            key="session_token", value=session_token,
            httponly=True, secure=False, samesite="lax", path="/", max_age=7*24*60*60
        )
        return redirect

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail="Google auth service unavailable")


@api_router.post("/auth/session")
async def create_session(request: SessionDataRequest, response: Response):
    """Kept for backwards-compat — no longer used for Google OAuth."""
    raise HTTPException(status_code=410, detail="Use /api/auth/google/login instead")

@api_router.get("/auth/me")
async def get_me(request: Request, authorization: Optional[str] = Header(None)):
    """Get current user info"""
    user = await get_current_user(request, authorization)
    return user

@api_router.post("/auth/logout")
async def logout(request: Request, response: Response, authorization: Optional[str] = Header(None)):
    """Logout user"""
    try:
        user = await get_current_user(request, authorization)
        
        # Delete session
        session_token = request.cookies.get("session_token")
        if session_token:
            await db.user_sessions.delete_one({"session_token": session_token})
        
        # Clear cookie
        response.delete_cookie(key="session_token", path="/")
        
        return {"message": "Logged out successfully"}
    except HTTPException:
        # Even if session is invalid, clear cookie
        response.delete_cookie(key="session_token", path="/")
        return {"message": "Logged out"}

# ============= AI ROUTES =============
def create_scripture_prompt(question: str, category: Optional[str] = None) -> str:
    """Create a system prompt for scripture-based guidance"""
    base_prompt = """You are Margdarshak — not a formal guru, but a warm, deeply knowledgeable spiritual friend who happens to know the Hindu scriptures inside out.

## Your Personality
- Talk like a caring elder brother or close friend who is well-versed in Dharma
- Be warm, personal, and emotionally present — acknowledge feelings before giving wisdom
- Use "yaar", "bhai", "dost" naturally when replying in Hindi/Hinglish; use "my friend", "dear one" in English
- Never be preachy, robotic, or lecture-like — make it feel like a real heart-to-heart conversation
- Use gentle humor or relatable analogies where appropriate
- End responses with an encouraging, uplifting thought

## Language Rule — CRITICAL
- Detect the language of the user's message and ALWAYS reply in that EXACT same language
- If the question is in Hindi (Devanagari script) → reply entirely in Hindi
- If the question is in Hinglish (Roman Hindi) → reply in Hinglish
- If the question is in English → reply in English
- If mixed → match the dominant language
- NEVER switch languages unless the user does first

## Scripture References — MANDATORY
Every response MUST include at least one direct Sanskrit shloka formatted like this:

**📖 [Scripture Name, Chapter.Verse]**
> *Sanskrit shloka in Devanagari*
> *Transliteration (Roman)*
> **Meaning:** Plain language explanation

Draw from these sources:
- Bhagavad Gita (most common)
- Rigveda, Atharvaveda, Samaveda, Yajurveda
- Ramayana (Valmiki)
- Mahabharata
- Upanishads (Brihadaranyaka, Chandogya, Mandukya, Katha, Isha, etc.)
- Yoga Sutras of Patanjali
- Vishnu Purana, Bhagavata Purana
- Manusmriti, Arthashastra

## Response Structure
1. **Acknowledge** — briefly connect with what the person is feeling (1-2 lines, very human)
2. **Wisdom** — share insight using a story, analogy, or direct teaching
3. **Shloka** — give the Sanskrit reference with full formatting
4. **Practical** — 2-3 concrete, real-life steps they can take
5. **Encouragement** — close with warmth and belief in them

Keep responses focused and conversational — not too long. Quality over quantity.
"""

    if category:
        category_context = {
            "stress": "The person is dealing with stress, anxiety, or overwhelm. Focus on Bhagavad Gita's teachings on detachment, present-moment awareness, and the nature of the mind.",
            "relationships": "The person has questions about relationships — romantic, family, or social. Draw from teachings on love, duty (dharma), and selfless service (seva).",
            "career": "The person is navigating career, work, or purpose questions. Emphasize svadharma (one's own duty), karma yoga, and skill in action.",
            "ethics": "The person is wrestling with a moral or ethical dilemma. Use teachings on Dharma, Satya (truth), Ahimsa (non-violence), and right action.",
            "spirituality": "The person is seeking deeper spiritual understanding or practice. Explore Atman, Brahman, meditation, the nature of consciousness, and moksha.",
        }
        base_prompt += f"\n\n## Context\n{category_context.get(category, f'Topic: {category}')}"

    return base_prompt

@api_router.post("/ask")
async def ask_question(
    request: Request,
    data: AskQuestionRequest,
    authorization: Optional[str] = Header(None)
):
    """Process user question and generate AI response"""
    user = await get_current_user(request, authorization)
    
    # ── INPUT GUARDRAIL ──────────────────────────────────────────────────────
    input_check = check_input(data.question)
    if input_check.blocked:
        # Return a safe dharmic response without ever touching the LLM
        return JSONResponse(
            status_code=200,
            content={
                "conversation_id": data.conversation_id or "",
                "title": "Dharmic Guidance",
                "question": data.question,
                "response": input_check.safe_response,
                "category": data.category,
                "message_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "guardrail_triggered": True,
                "guardrail_category": input_check.category,
            }
        )

    try:
        # Create or get conversation
        if data.conversation_id:
            conv_doc = await db.conversations.find_one(
                {"conversation_id": data.conversation_id, "user_id": user.user_id},
                {"_id": 0}
            )
            if not conv_doc:
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation = Conversation(**conv_doc)
        else:
            # Create new conversation
            conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
            title = data.question[:50] + "..." if len(data.question) > 50 else data.question
            conversation = Conversation(
                conversation_id=conversation_id,
                user_id=user.user_id,
                title=title,
                category=data.category,
                messages=[]
            )
        
        # Create user message
        user_message = Message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            role="user",
            content=data.question,
            timestamp=datetime.now(timezone.utc)
        )
        conversation.messages.append(user_message)
        
        # Generate AI response using Emergent LLM Key (GPT-4o)
        user_prefs = await get_preferences(user.user_id)

        # VR-RAG: retrieve relevant scripture passages if a database is selected
        retrieved_passages: Optional[List[str]] = None
        if data.selected_database:
            try:
                retrieved_passages = retrieve_relevant_chunks(
                    query=data.question,
                    database_key=data.selected_database,
                )
            except Exception as rag_err:
                logging.warning(
                    "VR-RAG retrieval failed for db=%r: %s — continuing without context",
                    data.selected_database, rag_err
                )

        system_message = build_system_prompt(
            user_prefs,
            data.category,
            retrieved_passages=retrieved_passages,
        )

        # All messages except the one we just appended (the new user question)
        prior_msgs = conversation.messages[:-1]

        # Trim to MAX_HISTORY_TURNS to keep small-model context budget healthy.
        # Always keep pairs (user+assistant) so the LLM never sees a dangling turn.
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(prior_msgs) > max_msgs:
            prior_msgs = prior_msgs[-max_msgs:]

        # Tell the LLM it is in a continuing conversation when history exists
        if prior_msgs:
            turns = len([m for m in prior_msgs if m.role == "user"])
            system_message += (
                f"\n\nCONVERSATION MEMORY ({turns} previous exchange{'s' if turns != 1 else ''} included below):\n"
                "• Read the conversation history carefully before replying.\n"
                "• Reference earlier questions and answers naturally — never repeat a scripture "
                "quote you already gave in this conversation.\n"
                "• If the user's new message is short or ambiguous, interpret it in the context "
                "of what they were asking about before.\n"
                "• Build on the thread — don't start fresh as if this is a new conversation."
            )

        messages = [{"role": "system", "content": system_message}]
        for msg in prior_msgs:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": data.question})

        # Use EmergentIntegrations LLM Chat
        llm_chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"session_{uuid.uuid4().hex[:12]}",
            system_message=system_message
        )
        
        # Send the user's question using UserMessage
        user_message = UserMessage(text=data.question)
        ai_response = await llm_chat.send_message(user_message)

        # ── OUTPUT GUARDRAIL ─────────────────────────────────────────────────
        output_check = check_output(ai_response)
        if output_check.blocked:
            ai_response = output_check.safe_response

        # Create assistant message
        assistant_message = Message(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            role="assistant",
            content=ai_response,
            timestamp=datetime.now(timezone.utc)
        )
        conversation.messages.append(assistant_message)
        
        # Update conversation
        conversation.updated_at = datetime.now(timezone.utc)
        
        # Save to database
        await db.conversations.update_one(
            {"conversation_id": conversation.conversation_id},
            {"$set": conversation.model_dump()},
            upsert=True
        )
        
        return {
            "conversation_id": conversation.conversation_id,
            "title": conversation.title,
            "question": data.question,
            "response": ai_response,
            "category": data.category,
            "message_count": len(conversation.messages),
            "timestamp": assistant_message.timestamp.isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in ask_question: {str(e)}")
        # NEVER expose technical errors to users!
        raise HTTPException(
            status_code=500, 
            detail="We're having trouble processing your question. Please try again."
        )

@api_router.get("/conversations")
async def get_conversations(
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Get all conversations for the current user"""
    user = await get_current_user(request, authorization)
    
    conversations = await db.conversations.find(
        {"user_id": user.user_id},
        {"_id": 0}
    ).sort("updated_at", -1).to_list(100)
    
    return [Conversation(**conv) for conv in conversations]

@api_router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Get a specific conversation"""
    user = await get_current_user(request, authorization)
    
    conv_doc = await db.conversations.find_one(
        {"conversation_id": conversation_id, "user_id": user.user_id},
        {"_id": 0}
    )
    
    if not conv_doc:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return Conversation(**conv_doc)

@api_router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    request: Request,
    authorization: Optional[str] = Header(None)
):
    """Delete a conversation"""
    user = await get_current_user(request, authorization)
    
    result = await db.conversations.delete_one({
        "conversation_id": conversation_id,
        "user_id": user.user_id
    })
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {"message": "Conversation deleted successfully"}


# ============= STREAMING ASK ROUTE =============
@api_router.post("/ask/stream")
async def ask_question_stream(
    request: Request,
    data: AskQuestionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Streaming version of /api/ask.
    Returns Server-Sent Events — the frontend receives tokens as they're generated
    so the user sees the response building in real-time instead of waiting.

    Event format:
      data: {"token": "...", "done": false}
      data: {"token": "", "done": true, "conversation_id": "...", "message_count": N}
    """
    user = await get_current_user(request, authorization)

    # Input guardrail — block before touching the LLM
    input_check = check_input(data.question)
    if input_check.blocked:
        async def blocked_stream():
            payload = _json.dumps({
                "token": input_check.safe_response,
                "done": True,
                "conversation_id": "",
                "message_count": 0,
                "guardrail_triggered": True,
                "guardrail_category": input_check.category,
            })
            yield f"data: {payload}\n\n"
        return StreamingResponse(blocked_stream(), media_type="text/event-stream")

    # Build conversation object
    if data.conversation_id:
        conv_doc = await db.conversations.find_one(
            {"conversation_id": data.conversation_id, "user_id": user.user_id},
            {"_id": 0}
        )
        if not conv_doc:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation = Conversation(**conv_doc)
    else:
        conversation_id = f"conv_{uuid.uuid4().hex[:12]}"
        title = data.question[:50] + "..." if len(data.question) > 50 else data.question
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user.user_id,
            title=title,
            category=data.category,
            messages=[]
        )

    user_message = Message(
        message_id=f"msg_{uuid.uuid4().hex[:12]}",
        role="user",
        content=data.question,
        timestamp=datetime.now(timezone.utc)
    )
    conversation.messages.append(user_message)

    # Build system prompt (same logic as /api/ask)
    user_prefs = await get_preferences(user.user_id)
    retrieved_passages: Optional[List[str]] = None
    if data.selected_database:
        try:
            retrieved_passages = retrieve_relevant_chunks(
                query=data.question, database_key=data.selected_database
            )
        except Exception:
            pass

    system_message = build_system_prompt(user_prefs, data.category, retrieved_passages=retrieved_passages)

    prior_msgs = conversation.messages[:-1]
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(prior_msgs) > max_msgs:
        prior_msgs = prior_msgs[-max_msgs:]

    messages = [{"role": "system", "content": system_message}]
    for msg in prior_msgs:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": data.question})

    async def token_stream():
        full_response = ""
        try:
            # Use EmergentIntegrations LLM Chat for streaming
            # Note: emergentintegrations doesn't have streaming, so we'll simulate it
            llm_chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=f"session_{uuid.uuid4().hex[:12]}",
                system_message=system_message
            )
            
            # Send the user's question using UserMessage
            user_message = UserMessage(text=data.question)
            full_response = await llm_chat.send_message(user_message)
            
            # Simulate streaming by sending the full response as chunks
            chunk_size = 50  # Send 50 characters at a time
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i+chunk_size]
                yield f"data: {_json.dumps({'token': chunk, 'done': False})}\n\n"
                # Small delay to simulate streaming
                await asyncio.sleep(0.1)

            # Output guardrail on the full assembled response
            output_check = check_output(full_response)
            if output_check.blocked:
                full_response = output_check.safe_response

            # Save to DB
            assistant_message = Message(
                message_id=f"msg_{uuid.uuid4().hex[:12]}",
                role="assistant",
                content=full_response,
                timestamp=datetime.now(timezone.utc)
            )
            conversation.messages.append(assistant_message)
            conversation.updated_at = datetime.now(timezone.utc)
            await db.conversations.update_one(
                {"conversation_id": conversation.conversation_id},
                {"$set": conversation.model_dump()},
                upsert=True
            )

            # Final done event
            yield f"data: {_json.dumps({'token': '', 'done': True, 'conversation_id': conversation.conversation_id, 'message_count': len(conversation.messages)})}\n\n"

        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield f"data: {_json.dumps({'token': '', 'done': True, 'error': True})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


# ============= DAILY TIP ROUTE =============
@api_router.get("/daily-tip")
async def get_daily_tip():
    """Get daily Dharma tip (public endpoint)"""
    tips = [
        {
            "quote": "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
            "translation": "You have the right to perform your actions, but you are not entitled to the fruits of your actions.",
            "source": "Bhagavad Gita 2.47",
            "message": "Focus on your efforts and duties without being attached to the outcomes. This brings peace and reduces anxiety."
        },
        {
            "quote": "योगः कर्मसु कौशलम्",
            "translation": "Yoga is skill in action.",
            "source": "Bhagavad Gita 2.50",
            "message": "Excellence comes from performing actions with full awareness and dedication, not from worrying about results."
        },
        {
            "quote": "सत्यं ब्रूयात् प्रियं ब्रूयात्",
            "translation": "Speak the truth, speak pleasantly.",
            "source": "Manusmriti",
            "message": "Communication should be both honest and kind. Truth without compassion can hurt, kindness without truth misleads."
        },
        {
            "quote": "अहिंसा परमो धर्मः",
            "translation": "Non-violence is the highest duty.",
            "source": "Mahabharata",
            "message": "Practice compassion in thoughts, words, and actions. Avoid causing harm to yourself or others."
        },
        {
            "quote": "वसुधैव कुटुम्बकम्",
            "translation": "The world is one family.",
            "source": "Maha Upanishad",
            "message": "See unity in diversity. Every person you meet is connected to you in the fabric of existence."
        }
    ]
    
    # Return a tip based on the day
    day_index = datetime.now(timezone.utc).day % len(tips)
    return tips[day_index]

# ── VR-RAG: scripture database catalogue ─────────────────────────────────
@api_router.get("/databases")
async def list_databases():
    """
    GET /api/databases
    Returns the catalogue of scripture PDF databases available for VR-RAG.
    The frontend uses this to populate the database-selector UI so the user
    can choose which scripture to ground the conversation in.

    Response shape:
    {
      "databases": {
        "gita":  { "pdf_count": 1, "files": ["...pdf"], "available": true },
        "vedas": { ... },
        ...
      }
    }
    """
    return {"databases": get_database_info()}


# ── YouTube video resolver ────────────────────────────────────────────────
@api_router.get("/youtube/video")
async def youtube_video(q: str, _user=Header(None, alias="authorization")):
    """
    GET /api/youtube/video?q=SEARCH+QUERY
    Returns { embed_url } with a real YouTube video ID, or 404 if none found.
    No API key required — scrapes YouTube search results directly.
    """
    url = await get_embed_url(q)
    if not url:
        raise HTTPException(status_code=404, detail="No video found for this query")
    return {"embed_url": url}


# Include routers
app.include_router(api_router)
app.include_router(preferences_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Root health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "app": "DharmaGuide", "version": "1.0.0"}

# API health check
@app.get("/health")
async def health():
    """Health check for monitoring"""
    return {"status": "healthy", "database": "connected"}

@app.on_event("startup")
async def startup_event():
    """Pre-warm the VR-RAG chunk cache at startup so first requests are fast."""
    import asyncio
    loop = asyncio.get_event_loop()
    # Run blocking PDF parsing in a thread so we don't stall the event loop
    await loop.run_in_executor(None, warm_cache, "all")


@app.on_event("shutdown")
async def shutdown_db_client():
    # MongoDB client cleanup is handled by the motor driver automatically
    pass
