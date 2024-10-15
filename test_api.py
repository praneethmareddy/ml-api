import requests

# Replace with your local URL (default is http://127.0.0.1:8000)
BASE_URL = "http://127.0.0.1:8000"

def test_update_model():
    url = f"{BASE_URL}/update_model"
    payload = {
        "text": "This is a test post to update the model.",
        "posted_by": "670edb0c857042d038804f4a"  # Replace with an actual UserID
    }
    
    response = requests.post(url, json=payload)
    
    print("Update Model Response:")
    print(response.status_code)
    print(response.json())

def test_recommend_posts():
    url = f"{BASE_URL}/recommend_posts"
    payload = {
        "user_id": "670edb0c857042d038804f4a",  # Replace with an actual UserID
        "top_n": 5
    }
    
    response = requests.post(url, json=payload)
    
    print("Recommend Posts Response:")
    print(response.status_code)
    print(response.json())

if __name__ == "__main__":
    test_update_model()
    test_recommend_posts()
