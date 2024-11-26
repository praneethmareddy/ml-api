from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bson.objectid import ObjectId

app = FastAPI()

# Allow all CORS requests (adjust this for production as needed)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB URI and database setup
MONGO_URI = 'mongodb+srv://praneethmareddy:saip9091@cluster0.rfvy63z.mongodb.net/thread?retryWrites=true&w=majority'
DATABASE_NAME = 'thread'
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
posts_collection = db['posts']
users_collection = db['users']

# Pydantic models for input validation
class UserInput(BaseModel):
    user_id: str
    top_n: int = 5

# Utility functions
def load_model():
    # Load the saved recommendation model from disk
    with open('recommendation_model.pkl', 'rb') as model_file:
        vectorizer, tfidf_matrix = pickle.load(model_file)
    return vectorizer, tfidf_matrix

def recommend_posts_for_user(user_id, top_n=5):
    # Load the trained model
    vectorizer, tfidf_matrix = load_model()

    # Ensure user exists in the database
    user = users_collection.find_one({'_id': ObjectId(user_id)}, {'following': 1})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Fetch the posts from the users that the current user is following
    following_ids = user.get('following', [])
    following_posts = list(posts_collection.find(
        {'postedBy': {'$in': following_ids}},
        {'postedBy': 1, 'text': 1, '_id': 1}
    ))

    # Fetch all posts to compare against
    all_posts = list(posts_collection.find({'postedBy': {'$ne': ObjectId(user_id)}}, {'postedBy': 1, 'text': 1, '_id': 1}))
    if len(all_posts) == 0:
        raise HTTPException(status_code=404, detail="No posts available for recommendation.")

    # Extract posts text for similarity comparison
    all_posts_text = [post['text'] for post in all_posts]
    following_posts_text = [post['text'] for post in following_posts]
    
    # Calculate similarity between the user's posts and all other posts
    user_posts_vector = vectorizer.transform(following_posts_text)
    all_posts_vector = vectorizer.transform(all_posts_text)
    similarities = cosine_similarity(user_posts_vector, all_posts_vector)
    
    # Calculate average similarity for each post
    avg_similarities = np.mean(similarities, axis=0)

    # Sort posts by similarity
    similar_indices = np.argsort(avg_similarities)[::-1]

    # Build the list of recommended posts
    similar_posts = []
    for i in similar_indices:
        post = all_posts[i]
        similar_posts.append({
            'postId': str(post['_id']),  # Add post ID
            'userId': str(post['postedBy']),
            'text': post['text']
        })

    # Remove duplicate posts
    unique_posts = []
    seen_posts = set()

    for post in similar_posts:
        post_tuple = (post['postId'], post['userId'], post['text'])  # Use postId for uniqueness
        if post_tuple not in seen_posts:
            seen_posts.add(post_tuple)
            unique_posts.append(post)

    # Combine following posts and recommended posts
    combined_posts = following_posts + unique_posts

    # Remove duplicates based on postId
    seen_posts = set()
    final_posts = []
    for post in combined_posts:
        if post['postId'] not in seen_posts:
            seen_posts.add(post['postId'])
            final_posts.append(post)

    # Return top N unique posts
    return final_posts[:top_n]

# FastAPI endpoint to get post recommendations
@app.post('/recommend_posts')
def recommend_posts(user_input: UserInput):
    recommendations = recommend_posts_for_user(user_input.user_id, user_input.top_n)
    return {"recommendations": recommendations}
