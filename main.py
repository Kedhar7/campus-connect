# main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import uvicorn
import datetime
from urllib.parse import parse_qs
import os

# Database imports
from sqlalchemy.orm import Session
from database import SessionLocal, Message

# ML imports for content moderation
from transformers import pipeline

# OAuth imports from Authlib
from authlib.integrations.starlette_client import OAuth, OAuthError

# Initialize the sentiment analysis pipeline with explicit PyTorch backend.
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f",
    framework="pt"
)

app = FastAPI(title="Campus Connect")

# Set a simple configuration dictionary so that Authlib does not conflict with FastAPI's configuration.
app.state.config = {}
oauth = OAuth(app)
oauth.config = app.state.config

# Enable CORS for development (adjust allowed origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at the root URL.
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# Dependency: Get a database session.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------------------------
# Simplified User Authentication (HTTP endpoints)
# ---------------------------
fake_users_db = {
    "student1@srm.edu.in": {
        "username": "student1@srm.edu.in",
        "full_name": "Student One",
        "password": "password1"
    },
    "student2@srm.edu.in": {
        "username": "student2@srm.edu.in",
        "full_name": "Student Two",
        "password": "password2"
    },
    # New Google users will be added dynamically.
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def fake_decode_token(token: str):
    # For this demo, the token is simply the user's email.
    return fake_users_db.get(token)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    user = fake_decode_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    email = form_data.username
    if not email.endswith("@srm.edu.in"):
        raise HTTPException(status_code=400, detail="Please sign in with your SRM email.")
    user = fake_users_db.get(email)
    if not user or user["password"] != form_data.password:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    # For demo, token is the email.
    return {"access_token": user["username"], "token_type": "bearer"}

# ---------------------------
# Google OAuth Configuration
# ---------------------------
# Replace the following with your actual Google OAuth credentials or set via environment variables.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "YOUR_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_GOOGLE_CLIENT_SECRET")

oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

@app.get("/auth/google")
async def auth_google(request: Request):
    # Convert the URL to a string explicitly.
    redirect_uri = str(request.url_for("auth_google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except OAuthError as error:
        raise HTTPException(status_code=400, detail=f"OAuth error: {error.error}")
    try:
        user_info = await oauth.google.parse_id_token(request, token)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing user information.")
    
    email = user_info.get("email")
    full_name = user_info.get("name")
    if not email.endswith("@srm.edu.in"):
        raise HTTPException(status_code=400, detail="Only SRM email accounts are allowed.")
    if email not in fake_users_db:
        fake_users_db[email] = {"username": email, "full_name": full_name, "password": None}
    response = RedirectResponse(url=f"/?token={email}")
    response.set_cookie(key="access_token", value=email)
    return response

# ---------------------------
# Pydantic Model for Chat Messages
# ---------------------------
class ChatMessage(BaseModel):
    sender: str
    content: str
    timestamp: str

# In-memory storage for chat messages (optional).
chat_messages = []

# ---------------------------
# Enhanced Content Moderation using ML
# ---------------------------
def moderate_message(content: str) -> bool:
    banned_words = ["spam", "scam", "advertisement"]
    lower_content = content.lower()
    for word in banned_words:
        if word in lower_content:
            return True
    result = sentiment_pipeline(content)
    sentiment = result[0]
    if sentiment["label"] == "NEGATIVE" and sentiment["score"] > 0.95:
        return True
    return False

# ---------------------------
# WebSocket Connection Manager
# ---------------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

# ---------------------------
# WebSocket Endpoint for Real-Time Chat
# ---------------------------
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    query_params = parse_qs(websocket.scope["query_string"].decode())
    token_list = query_params.get("token")
    if not token_list:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    token = token_list[0]
    current_user = fake_decode_token(token)
    if not current_user:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket)
    db: Session = SessionLocal()  # Create a database session.
    try:
        while True:
            data = await websocket.receive_json()
            sender = current_user["full_name"]
            content = data.get("content", "")
            timestamp = datetime.datetime.utcnow()
            
            if moderate_message(content):
                await websocket.send_json({"error": "Message flagged as inappropriate."})
                continue

            message_entry = {
                "sender": sender,
                "content": content,
                "timestamp": timestamp.isoformat() + "Z"
            }
            chat_messages.append(message_entry)
            
            db_message = Message(sender=sender, content=content, timestamp=timestamp)
            db.add(db_message)
            db.commit()
            db.refresh(db_message)
            
            await manager.broadcast(message_entry)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    finally:
        db.close()

# ---------------------------
# REST API Endpoint for Searching Chat History (Database-backed)
# ---------------------------
@app.get("/search")
async def search_messages(keyword: str, current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    results = db.query(Message).filter(Message.content.ilike(f"%{keyword}%")).all()
    formatted_results = [{
        "sender": msg.sender,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat() + "Z"
    } for msg in results]
    return {"results": formatted_results}

# ---------------------------
# Run the Application
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
