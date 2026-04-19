"""
Backend API Tests for Margdarshak/Bhakti Spiritual Guidance App
Tests all endpoints specified in the testing request
"""
import pytest
import requests
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load frontend .env to get BACKEND_URL
frontend_env = Path(__file__).parent.parent.parent / 'frontend' / '.env'
if frontend_env.exists():
    load_dotenv(frontend_env)

BASE_URL = os.environ.get('EXPO_PUBLIC_BACKEND_URL')
if not BASE_URL:
    raise ValueError("EXPO_PUBLIC_BACKEND_URL not found in environment")

BASE_URL = BASE_URL.rstrip('/')

# Test credentials from /app/memory/test_credentials.md
TEST_USER_NAME = "Test User"
TEST_USER_EMAIL = "test@test.com"


@pytest.fixture(scope="session")
def auth_session():
    """Create authenticated session using dev-login"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    
    # Dev login
    resp = session.post(
        f"{BASE_URL}/api/auth/dev-login",
        json={"name": TEST_USER_NAME, "email": TEST_USER_EMAIL}
    )
    assert resp.status_code == 200, f"Dev login failed: {resp.status_code} {resp.text}"
    
    data = resp.json()
    token = data.get("session_token")
    assert token, "session_token missing in dev-login response"
    
    # Set Bearer token
    session.headers.update({"Authorization": f"Bearer {token}"})
    session._test_token = token
    session._test_user_id = data.get("user_id")
    
    print(f"\n✓ Authenticated as: {data.get('name')} ({data.get('email')})")
    print(f"  User ID: {data.get('user_id')}")
    print(f"  Session token: {token[:20]}...")
    
    return session


@pytest.fixture
def anon_session():
    """Unauthenticated session"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestPublicEndpoints:
    """Test public endpoints that don't require authentication"""
    
    def test_daily_tip_returns_sanskrit_quote(self, anon_session):
        """GET /api/daily-tip returns Sanskrit quotes"""
        response = anon_session.get(f"{BASE_URL}/api/daily-tip")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "quote" in data, "Missing 'quote' field"
        assert "translation" in data, "Missing 'translation' field"
        assert "source" in data, "Missing 'source' field"
        assert "message" in data, "Missing 'message' field"
        
        # Verify it's actually Sanskrit (Devanagari script)
        assert any('\u0900' <= c <= '\u097F' for c in data["quote"]), \
            "Quote should contain Devanagari (Sanskrit) characters"
        
        # Verify source is a real scripture
        known_sources = ["Bhagavad Gita", "Mahabharata", "Manusmriti", "Upanishad"]
        assert any(s in data["source"] for s in known_sources), \
            f"Source '{data['source']}' doesn't match known scriptures"
        
        print(f"✓ Daily tip endpoint working")
        print(f"  Quote: {data['quote']}")
        print(f"  Source: {data['source']}")


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_dev_login_creates_session_and_returns_token(self, anon_session):
        """POST /api/auth/dev-login creates session and returns session_token"""
        test_email = f"test_login_{int(time.time())}@test.com"
        
        response = anon_session.post(
            f"{BASE_URL}/api/auth/dev-login",
            json={"name": "Login Test User", "email": test_email}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "session_token" in data, "Missing 'session_token' in response"
        assert "user_id" in data, "Missing 'user_id' in response"
        assert "email" in data, "Missing 'email' in response"
        assert "name" in data, "Missing 'name' in response"
        
        assert data["email"] == test_email, "Email mismatch"
        assert data["name"] == "Login Test User", "Name mismatch"
        assert data["session_token"].startswith("dev_session_"), \
            "Session token should start with 'dev_session_'"
        
        print(f"✓ Dev login endpoint working")
        print(f"  Created user: {data['name']} ({data['email']})")
        print(f"  Session token: {data['session_token'][:30]}...")
    
    def test_auth_me_with_bearer_token_returns_user(self, auth_session):
        """GET /api/auth/me with Bearer token returns user"""
        response = auth_session.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "user_id" in data, "Missing 'user_id'"
        assert "email" in data, "Missing 'email'"
        assert "name" in data, "Missing 'name'"
        
        assert data["email"] == TEST_USER_EMAIL, f"Expected {TEST_USER_EMAIL}, got {data['email']}"
        assert data["name"] == TEST_USER_NAME, f"Expected {TEST_USER_NAME}, got {data['name']}"
        
        print(f"✓ Auth me endpoint working with Bearer token")
        print(f"  User: {data['name']} ({data['email']})")
    
    def test_auth_me_without_token_returns_401(self, anon_session):
        """GET /api/auth/me without token returns 401"""
        response = anon_session.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Auth me correctly returns 401 without token")


class TestAskEndpoint:
    """Test AI question endpoint with GPT-4o"""
    
    def test_ask_with_bearer_token_generates_ai_response(self, auth_session):
        """POST /api/ask with Bearer token generates AI response using GPT-4o"""
        question_data = {
            "question": "What does the Bhagavad Gita say about inner peace?",
            "category": "stress"
        }
        
        response = auth_session.post(f"{BASE_URL}/api/ask", json=question_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "conversation_id" in data, "Missing 'conversation_id'"
        assert "question" in data, "Missing 'question'"
        assert "response" in data, "Missing 'response'"
        assert "category" in data, "Missing 'category'"
        assert "timestamp" in data, "Missing 'timestamp'"
        
        assert data["question"] == question_data["question"], "Question mismatch"
        assert data["category"] == question_data["category"], "Category mismatch"
        assert len(data["response"]) > 100, f"AI response too short: {len(data['response'])} chars"
        
        # Verify conversation_id format
        assert data["conversation_id"].startswith("conv_"), \
            f"Conversation ID should start with 'conv_', got {data['conversation_id']}"
        
        print(f"✓ Ask endpoint working with GPT-4o")
        print(f"  Conversation ID: {data['conversation_id']}")
        print(f"  Response length: {len(data['response'])} chars")
        print(f"  Response preview: {data['response'][:150]}...")
        
        # Store for cleanup
        pytest.test_conv_id = data["conversation_id"]
        
        # Wait for DB write
        time.sleep(0.5)
    
    def test_ask_without_auth_returns_401(self, anon_session):
        """POST /api/ask without auth returns 401"""
        response = anon_session.post(
            f"{BASE_URL}/api/ask",
            json={"question": "Test question"}
        )
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Ask endpoint correctly returns 401 without auth")


class TestConversationsEndpoints:
    """Test conversation management endpoints"""
    
    def test_get_conversations_with_bearer_token_returns_list(self, auth_session):
        """GET /api/conversations with Bearer token returns conversation list"""
        response = auth_session.get(f"{BASE_URL}/api/conversations")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, list), "Response should be a list"
        
        if len(data) > 0:
            conv = data[0]
            assert "conversation_id" in conv, "Conversation missing 'conversation_id'"
            assert "title" in conv, "Conversation missing 'title'"
            assert "user_id" in conv, "Conversation missing 'user_id'"
            assert "messages" in conv, "Conversation missing 'messages'"
            assert "created_at" in conv, "Conversation missing 'created_at'"
            assert "updated_at" in conv, "Conversation missing 'updated_at'"
            
            print(f"✓ Get conversations endpoint working")
            print(f"  Found {len(data)} conversations")
            print(f"  Latest: {conv['title'][:50]}...")
        else:
            print("✓ Get conversations endpoint working (empty list)")
    
    def test_get_conversations_without_auth_returns_401(self, anon_session):
        """GET /api/conversations without auth returns 401"""
        response = anon_session.get(f"{BASE_URL}/api/conversations")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Conversations endpoint correctly returns 401 without auth")


class TestPreferencesEndpoint:
    """Test preferences/onboarding schema endpoint"""
    
    def test_preferences_schema_returns_4_onboarding_steps(self, anon_session):
        """GET /api/preferences/schema returns 4 onboarding steps"""
        response = anon_session.get(f"{BASE_URL}/api/preferences/schema")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "steps" in data, "Missing 'steps' field"
        
        steps = data["steps"]
        assert len(steps) == 4, f"Expected 4 onboarding steps, got {len(steps)}"
        
        # Verify each step has required fields
        for i, step in enumerate(steps):
            assert "id" in step, f"Step {i} missing 'id'"
            assert "title" in step, f"Step {i} missing 'title'"
            assert "description" in step, f"Step {i} missing 'description'"
        
        print(f"✓ Preferences schema endpoint working")
        print(f"  Onboarding steps: {len(steps)}")
        for step in steps:
            print(f"    - {step['title']}")


class TestDatabasesEndpoint:
    """Test scripture database catalog endpoint"""
    
    def test_databases_returns_scripture_catalog(self, anon_session):
        """GET /api/databases returns scripture database catalog"""
        response = anon_session.get(f"{BASE_URL}/api/databases")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "databases" in data, "Missing 'databases' field"
        
        dbs = data["databases"]
        
        # Verify core database keys exist
        required_keys = ["gita", "vedas", "upanishads", "mahabharata", "puran", "all"]
        for key in required_keys:
            assert key in dbs, f"Missing database key: {key}"
        
        # Verify each database has required structure
        for db_key, db_info in dbs.items():
            assert "pdf_count" in db_info, f"Database '{db_key}' missing 'pdf_count'"
            assert "files" in db_info, f"Database '{db_key}' missing 'files'"
            assert "available" in db_info, f"Database '{db_key}' missing 'available'"
            assert isinstance(db_info["files"], list), f"Database '{db_key}' files should be a list"
        
        print(f"✓ Databases endpoint working")
        print(f"  Available databases: {len(dbs)}")
        for db_key, db_info in dbs.items():
            print(f"    - {db_key}: {db_info['pdf_count']} PDFs, available={db_info['available']}")


class TestDataPersistence:
    """Test data persistence in MongoDB"""
    
    def test_conversation_persists_in_database(self, auth_session):
        """Create conversation → GET to verify persistence"""
        # Create conversation
        question_data = {
            "question": "TEST_PERSISTENCE: What is karma yoga?",
            "category": "spirituality"
        }
        
        create_response = auth_session.post(f"{BASE_URL}/api/ask", json=question_data)
        assert create_response.status_code == 200, "Failed to create conversation"
        
        conv_id = create_response.json()["conversation_id"]
        time.sleep(0.5)  # Wait for DB write
        
        # Verify persistence by fetching
        get_response = auth_session.get(f"{BASE_URL}/api/conversations/{conv_id}")
        assert get_response.status_code == 200, "Failed to retrieve conversation"
        
        data = get_response.json()
        assert data["conversation_id"] == conv_id, "Conversation ID mismatch"
        assert "TEST_PERSISTENCE" in data["messages"][0]["content"], \
            "Question not persisted correctly"
        
        # Verify message structure
        assert len(data["messages"]) >= 2, "Should have at least user + assistant messages"
        
        user_msg = data["messages"][0]
        assert user_msg["role"] == "user", "First message should be from user"
        assert user_msg["content"] == question_data["question"], "User message mismatch"
        
        assistant_msg = data["messages"][1]
        assert assistant_msg["role"] == "assistant", "Second message should be from assistant"
        assert len(assistant_msg["content"]) > 50, "Assistant response too short"
        
        print(f"✓ Data persistence verified")
        print(f"  Conversation saved and retrieved: {conv_id}")
        print(f"  Messages: {len(data['messages'])}")
        
        # Cleanup
        auth_session.delete(f"{BASE_URL}/api/conversations/{conv_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", f"--junitxml=/app/test_reports/pytest/margdarshak_api_results.xml"])
