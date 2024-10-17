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
users_collection = db['users']  # Assuming there's a collection for valid users

# Pydantic models for input validation
class PostInput(BaseModel):
    text: str
    posted_by: str

class UserInput(BaseModel):
    user_id: str
    top_n: int = 5

# Utility functions
def get_existing_posts():
    # Fetch all posts from the collection
    posts = posts_collection.find({}, {'postedBy': 1, 'text': 1, '_id': 1})
    return [(post['_id'], post['postedBy'], post['text']) for post in posts]

def update_model(new_post, posted_by):
    # Update the recommendation model with the new post
    existing_posts = get_existing_posts()
    existing_posts.append((None, posted_by, new_post))  # None for the new post ID

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text for _, _, text in existing_posts])

    with open('recommendation_model.pkl', 'wb') as model_file:
        pickle.dump((vectorizer, tfidf_matrix), model_file)

def load_model():
    # Load the saved recommendation model from disk
    with open('recommendation_model.pkl', 'rb') as model_file:
        vectorizer, tfidf_matrix = pickle.load(model_file)
    return vectorizer, tfidf_matrix

def recommend_posts_for_user(user_id, top_n=5):
    # Load the trained model
    vectorizer, tfidf_matrix = load_model()

    # Fetch the user's posts
    try:
        user_posts = list(posts_collection.find({'postedBy': ObjectId(user_id)}, {'postedBy': 1, 'text': 1, '_id': 1}))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error fetching posts: " + str(e))

    if not user_posts:
        raise HTTPException(status_code=404, detail="No posts found for the user.")

    # Extract user posts text for similarity comparison
    user_posts_text = [post['text'] for post in user_posts]
    user_posts_vector = vectorizer.transform(user_posts_text)

    # Fetch all posts to compare against
    all_posts = list(posts_collection.find({'postedBy': {'$ne': ObjectId(user_id)}}, {'postedBy': 1, 'text': 1, '_id': 1}))

    if len(all_posts) == 0:
        raise HTTPException(status_code=404, detail="No posts available for recommendation.")

    all_posts_text = [post['text'] for post in all_posts]

    # Calculate similarity between the user's posts and all other posts
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

    # Check if the users who posted are valid
    valid_user_ids = {str(user['_id']) for user in users_collection.find({}, {'_id': 1})}
    
    # Filter out posts from invalid users
    filtered_posts = [post for post in unique_posts if post['userId'] in valid_user_ids]

    return filtered_posts[:top_n]

# FastAPI endpoints
@app.post('/update_model')
def update_model_endpoint(post_input: PostInput):
    # Endpoint to update the model with a new post
    update_model(post_input.text, post_input.posted_by)
    return {"message": "Model updated with the new post."}

@app.post('/recommend_posts')
def recommend_posts(user_input: UserInput):
    # Endpoint to get post recommendations for a user
    recommendations = recommend_posts_for_user(user_input.user_id, user_input.top_n)
    return {"recommendations": recommendations}
