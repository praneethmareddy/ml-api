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
# Collection for user relationships (e.g., following/followers)
users_collection = db['users']

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
    posts = posts_collection.find({}, {'postedBy': 1, 'text': 1})
    return [(post['postedBy'], post['text']) for post in posts]

def update_model(new_post, posted_by):
    # Update the recommendation model with the new post
    existing_posts = get_existing_posts()
    existing_posts.append((posted_by, new_post))

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text for _, text in existing_posts])

    with open('recommendation_model.pkl', 'wb') as model_file:
        pickle.dump((vectorizer, tfidf_matrix), model_file)

def load_model():
    # Load the saved recommendation model from disk
    with open('recommendation_model.pkl', 'rb') as model_file:
        vectorizer, tfidf_matrix = pickle.load(model_file)
    return vectorizer, tfidf_matrix

def get_following_users(user_id):
    # Fetch the users that the specified user is following
    user = users_collection.find_one({'_id': ObjectId(user_id)}, {'following': 1})
    if user and 'following' in user:
        return user['following']
    return []

def recommend_posts_for_user(user_id, top_n=5):
    # Load the trained model
    vectorizer, tfidf_matrix = load_model()

    # Fetch the user's posts
    try:
        user_posts = list(posts_collection.find({'postedBy': ObjectId(user_id)}, {'postedBy': 1, 'text': 1}))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error fetching posts: " + str(e))

    if not user_posts:
        raise HTTPException(status_code=404, detail="No posts found for the user.")

    # Extract user posts text for similarity comparison
    user_posts_text = [post['text'] for post in user_posts]
    user_posts_vector = vectorizer.transform(user_posts_text)

    # Fetch all posts to compare against
    all_posts = list(posts_collection.find({}, {'postedBy': 1, 'text': 1}))

    if len(all_posts) == 0:
        raise HTTPException(status_code=404, detail="No posts available for recommendation.")

    # Get the users that the current user is following
    following_users = get_following_users(user_id)

    # Filter out posts from users that the current user is following
    filtered_posts = [post for post in all_posts if str(post['postedBy']) not in following_users]
    if not filtered_posts:
        raise HTTPException(status_code=404, detail="No available posts for recommendation outside of followed users.")

    # Prepare for similarity calculation
    filtered_posts_text = [post['text'] for post in filtered_posts]
    all_posts_vector = vectorizer.transform(filtered_posts_text)
    
    # Calculate similarity between the user's posts and all other posts
    similarities = cosine_similarity(user_posts_vector, all_posts_vector)
    
    # Calculate average similarity for each filtered post
    avg_similarities = np.mean(similarities, axis=0)

    # Sort posts by similarity
    similar_indices = np.argsort(avg_similarities)[::-1]

    # Build the list of recommended posts
    unique_posts = []
    seen_posts = set()

    for i in similar_indices:
        post = filtered_posts[i]
        post_tuple = (str(post['postedBy']), post['text'])
        if post_tuple not in seen_posts:
            seen_posts.add(post_tuple)
            unique_posts.append({
                'userId': str(post['postedBy']),
                'text': post['text']
            })
        if len(unique_posts) == top_n:  # Limit to top_n recommendations
            break

    return unique_posts

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

@app.delete('/delete_post/{post_id}')
def delete_post(post_id: str):
    # Endpoint to delete a post by its ID
    result = posts_collection.delete_one({'_id': ObjectId(post_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Post not found.")
    return {"message": "Post deleted successfully."}
