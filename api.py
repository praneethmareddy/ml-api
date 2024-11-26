from fastapi import HTTPException
from pymongo import MongoClient
import numpy as np
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# MongoDB connection setup
MONGO_URI = 'mongodb+srv://praneethmareddy:saip9091@cluster0.rfvy63z.mongodb.net/thread?retryWrites=true&w=majority'
DATABASE_NAME = 'thread'
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
posts_collection = db['posts']
users_collection = db['users']

# Utility function to load the trained recommendation model
def load_model():
    # Load the saved recommendation model from disk (assumed to be a pickle file)
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

    # Fetch the user's posts
    user_posts = list(posts_collection.find({'postedBy': ObjectId(user_id)}, {'postedBy': 1, 'text': 1, '_id': 1}))
    if not user_posts:
        raise HTTPException(status_code=404, detail="No posts found for the user.")

    # Extract user posts' text for similarity comparison
    user_posts_text = [post['text'] for post in user_posts]
    user_posts_vector = vectorizer.transform(user_posts_text)

    # Fetch posts from users the current user is following
    following_ids = user.get('following', [])
    following_posts = list(posts_collection.find(
        {'postedBy': {'$in': following_ids}},
        {'postedBy': 1, 'text': 1, '_id': 1}
    ))

    # Ensure all following posts are valid (posts from valid users)
    valid_following_posts = []
    for post in following_posts:
        if users_collection.find_one({'_id': post['postedBy']}):  # Check if `postedBy` user exists
            valid_following_posts.append({
                'postId': str(post['_id']),
                'userId': str(post['postedBy']),
                'text': post['text']
            })

    # Fetch all other posts for recommendations (posts not from the current user)
    other_posts = list(posts_collection.find(
        {'postedBy': {'$ne': ObjectId(user_id)}},
        {'postedBy': 1, 'text': 1, '_id': 1}
    ))

    # Calculate similarities between user's posts and other posts
    all_posts_text = [post['text'] for post in other_posts]
    all_posts_vector = vectorizer.transform(all_posts_text)
    similarities = cosine_similarity(user_posts_vector, all_posts_vector)
    avg_similarities = np.mean(similarities, axis=0)
    similar_indices = np.argsort(avg_similarities)[::-1]  # Sort by similarity, highest first

    # Extract recommended posts based on similarity
    recommended_posts = []
    for i in similar_indices:
        post = other_posts[i]
        # Check if both post and user are valid
        if (
            posts_collection.find_one({'_id': post['_id']}) and  # Check if post exists
            users_collection.find_one({'_id': post['postedBy']})  # Check if user exists
        ):
            recommended_posts.append({
                'postId': str(post['_id']),
                'userId': str(post['postedBy']),
                'text': post['text']
            })

    # Combine valid following posts and recommended posts
    combined_posts = valid_following_posts + recommended_posts

    # Deduplicate posts based on postId
    seen_posts = set()
    unique_posts = []
    for post in combined_posts:
        if post['postId'] not in seen_posts:
            seen_posts.add(post['postId'])
            unique_posts.append(post)

    # Return top N unique posts
    return unique_posts[:top_n]
