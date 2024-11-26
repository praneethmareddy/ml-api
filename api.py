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

    # Fetch posts from following users
    following_ids = user.get('following', [])
    following_posts = list(posts_collection.find(
        {'postedBy': {'$in': following_ids}},
        {'postedBy': 1, 'text': 1, '_id': 1}
    ))

    # Ensure all following posts are valid
    valid_following_posts = []
    for post in following_posts:
        if users_collection.find_one({'_id': post['postedBy']}):  # Check if `postedBy` user exists
            valid_following_posts.append({
                'postId': str(post['_id']),
                'userId': str(post['postedBy']),
                'text': post['text']
            })

    # Fetch all other posts for recommendations
    other_posts = list(posts_collection.find(
        {'postedBy': {'$ne': ObjectId(user_id)}},
        {'postedBy': 1, 'text': 1, '_id': 1}
    ))

    # Calculate similarities
    all_posts_text = [post['text'] for post in other_posts]
    all_posts_vector = vectorizer.transform(all_posts_text)
    similarities = cosine_similarity(user_posts_vector, all_posts_vector)
    avg_similarities = np.mean(similarities, axis=0)
    similar_indices = np.argsort(avg_similarities)[::-1]

    # Extract recommended posts
    recommended_posts = []
    for i in similar_indices:
        post = other_posts[i]
        # Check if both post and user are valid
        if (
            posts_collection.find_one({'_id': post['_id']}) and  # Check if post exists
            users_collection.find_one({'_id': post['postedBy']})  # Check if user is valid
        ):
            recommended_posts.append({
                'postId': str(post['_id']),
                'userId': str(post['postedBy']),
                'text': post['text']
            })

    # Combine following posts and recommended posts
    combined_posts = valid_following_posts + recommended_posts

    # Deduplicate posts
    seen_posts = set()
    unique_posts = []
    for post in combined_posts:
        if post['postId'] not in seen_posts:
            seen_posts.add(post['postId'])
            unique_posts.append(post)

    return unique_posts[:top_n]
