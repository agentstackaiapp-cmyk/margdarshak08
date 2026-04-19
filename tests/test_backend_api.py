"""
Backend API Tests for DharmaGuide
Tests: Auth endpoints, Daily tip, Ask question, Conversations CRUD
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

BASE_URL = os.environ.get('EXPO_PUBLIC_BACKEND_URL', 'https://dharma-guide-20.preview.emergentagent.com').rstrip('/')
SESSION_TOKEN = "test_session_frontend_1775878799661"

@pytest.fixture
def api_client():
    """Shared requests session with auth"""
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
    })
    session.cookies.set("session_token", SESSION_TOKEN, domain=BASE_URL.replace("https://", "").replace("http://", ""))
    return session

@pytest.fixture
def api_client_no_auth():
    """Requests session without auth"""
    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})
    return session


class TestPublicEndpoints:
    """Test public endpoints that don't require authentication"""
    
    def test_daily_tip_endpoint(self, api_client_no_auth):
        """Test daily tip endpoint returns valid data"""
        response = api_client_no_auth.get(f"{BASE_URL}/api/daily-tip")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "quote" in data, "Response missing 'quote' field"
        assert "translation" in data, "Response missing 'translation' field"
        assert "source" in data, "Response missing 'source' field"
        assert "message" in data, "Response missing 'message' field"
        print(f"✓ Daily tip endpoint working - Quote: {data['quote'][:50]}...")


class TestAuthEndpoints:
    """Test authentication endpoints"""
    
    def test_auth_me_with_valid_session(self, api_client):
        """Test /api/auth/me with valid session token"""
        response = api_client.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "user_id" in data, "Response missing 'user_id'"
        assert "email" in data, "Response missing 'email'"
        assert "name" in data, "Response missing 'name'"
        assert data["user_id"] == "test-user-frontend", f"Expected test-user-frontend, got {data['user_id']}"
        assert data["email"] == "testuser@dharma.com", f"Expected testuser@dharma.com, got {data['email']}"
        print(f"✓ Auth me endpoint working - User: {data['name']}")
    
    def test_auth_me_without_session(self, api_client_no_auth):
        """Test /api/auth/me without session returns 401"""
        response = api_client_no_auth.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Auth me correctly returns 401 without session")


class TestAskEndpoint:
    """Test AI question asking endpoint"""
    
    def test_ask_question_creates_conversation(self, api_client):
        """Test asking a question creates conversation and returns AI response"""
        question_data = {
            "question": "How can I find inner peace in stressful times?",
            "category": "stress"
        }
        
        response = api_client.post(f"{BASE_URL}/api/ask", json=question_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        
        data = response.json()
        assert "conversation_id" in data, "Response missing 'conversation_id'"
        assert "question" in data, "Response missing 'question'"
        assert "response" in data, "Response missing 'response'"
        assert "category" in data, "Response missing 'category'"
        assert data["question"] == question_data["question"], "Question mismatch"
        assert data["category"] == question_data["category"], "Category mismatch"
        assert len(data["response"]) > 50, "AI response too short"
        
        print(f"✓ Ask endpoint working - Conversation ID: {data['conversation_id']}")
        print(f"  AI response length: {len(data['response'])} chars")
        
        # Store conversation_id for later tests
        pytest.test_conversation_id = data["conversation_id"]
        
        # Wait a bit for AI response to be saved
        time.sleep(1)
    
    def test_ask_question_without_auth(self, api_client_no_auth):
        """Test asking question without auth returns 401"""
        question_data = {
            "question": "Test question",
            "category": "stress"
        }
        
        response = api_client_no_auth.post(f"{BASE_URL}/api/ask", json=question_data)
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Ask endpoint correctly returns 401 without auth")


class TestConversationsEndpoints:
    """Test conversation CRUD endpoints"""
    
    def test_get_conversations_list(self, api_client):
        """Test getting list of conversations"""
        response = api_client.get(f"{BASE_URL}/api/conversations")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert isinstance(data, list), "Response should be a list"
        
        if len(data) > 0:
            conv = data[0]
            assert "conversation_id" in conv, "Conversation missing 'conversation_id'"
            assert "title" in conv, "Conversation missing 'title'"
            assert "messages" in conv, "Conversation missing 'messages'"
            assert "user_id" in conv, "Conversation missing 'user_id'"
            print(f"✓ Get conversations working - Found {len(data)} conversations")
        else:
            print("✓ Get conversations working - Empty list (no conversations yet)")
    
    def test_get_specific_conversation(self, api_client):
        """Test getting a specific conversation by ID"""
        # First create a conversation
        question_data = {
            "question": "What is dharma?",
            "category": "spirituality"
        }
        
        create_response = api_client.post(f"{BASE_URL}/api/ask", json=question_data)
        assert create_response.status_code == 200, "Failed to create conversation"
        
        conversation_id = create_response.json()["conversation_id"]
        time.sleep(1)  # Wait for save
        
        # Now get the conversation
        response = api_client.get(f"{BASE_URL}/api/conversations/{conversation_id}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data["conversation_id"] == conversation_id, "Conversation ID mismatch"
        assert "messages" in data, "Response missing 'messages'"
        assert len(data["messages"]) >= 2, "Should have at least user and assistant messages"
        
        # Verify message structure
        user_msg = data["messages"][0]
        assert user_msg["role"] == "user", "First message should be from user"
        assert user_msg["content"] == question_data["question"], "User message content mismatch"
        
        assistant_msg = data["messages"][1]
        assert assistant_msg["role"] == "assistant", "Second message should be from assistant"
        assert len(assistant_msg["content"]) > 50, "Assistant response too short"
        
        print(f"✓ Get specific conversation working - ID: {conversation_id}")
        print(f"  Messages: {len(data['messages'])}, Title: {data['title'][:50]}...")
        
        # Store for delete test
        pytest.test_delete_conversation_id = conversation_id
    
    def test_get_nonexistent_conversation(self, api_client):
        """Test getting a conversation that doesn't exist returns 404"""
        response = api_client.get(f"{BASE_URL}/api/conversations/nonexistent_id_12345")
        assert response.status_code == 404, f"Expected 404, got {response.status_code}"
        print("✓ Get nonexistent conversation correctly returns 404")
    
    def test_delete_conversation(self, api_client):
        """Test deleting a conversation"""
        # Use conversation created in previous test
        if not hasattr(pytest, 'test_delete_conversation_id'):
            pytest.skip("No conversation ID available for delete test")
        
        conversation_id = pytest.test_delete_conversation_id
        
        # Delete the conversation
        response = api_client.delete(f"{BASE_URL}/api/conversations/{conversation_id}")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert "message" in data, "Response missing 'message'"
        
        # Verify it's deleted by trying to get it
        get_response = api_client.get(f"{BASE_URL}/api/conversations/{conversation_id}")
        assert get_response.status_code == 404, "Conversation should be deleted"
        
        print(f"✓ Delete conversation working - Deleted ID: {conversation_id}")
    
    def test_conversations_without_auth(self, api_client_no_auth):
        """Test conversations endpoints without auth return 401"""
        response = api_client_no_auth.get(f"{BASE_URL}/api/conversations")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("✓ Conversations endpoint correctly returns 401 without auth")


class TestDataPersistence:
    """Test that data is actually persisted in database"""
    
    def test_create_and_verify_persistence(self, api_client):
        """Test Create → GET pattern to verify data persistence"""
        # Create a conversation
        question_data = {
            "question": "TEST_PERSISTENCE: How to practice meditation?",
            "category": "spirituality"
        }
        
        create_response = api_client.post(f"{BASE_URL}/api/ask", json=question_data)
        assert create_response.status_code == 200, "Failed to create conversation"
        
        conversation_id = create_response.json()["conversation_id"]
        time.sleep(1)  # Wait for database write
        
        # Verify persistence by fetching from database
        get_response = api_client.get(f"{BASE_URL}/api/conversations/{conversation_id}")
        assert get_response.status_code == 200, "Failed to retrieve conversation"
        
        data = get_response.json()
        assert data["conversation_id"] == conversation_id, "Conversation ID mismatch"
        assert data["user_id"] == "test-user-frontend", "User ID mismatch"
        assert "TEST_PERSISTENCE" in data["messages"][0]["content"], "Question not persisted correctly"
        
        print(f"✓ Data persistence verified - Conversation saved and retrieved correctly")
        
        # Cleanup
        api_client.delete(f"{BASE_URL}/api/conversations/{conversation_id}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
