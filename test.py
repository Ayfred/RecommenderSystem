import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample user data from JSON
user_data = """
{
  "user_id": "123456789",
  "username": "example_user",
  "email": "user@example.com",
  "password_hash": "hashed_password_here",
  "profile_picture": "https://example.com/profile_picture.jpg",
  "date_of_birth": "1990-01-01",
  "location": "New York, USA",
  "preferred_categories": ["Politics", "Technology", "Sports"],
  "reading_history": [],
  "saved_articles": [],
  "subscription_status": "active",
  "notification_preferences": {
    "email": true,
    "push_notifications": true
  },
  "user_preferences": {
    "Politics": 4,
    "Technology": 3,
    "Business": 5,
    "Sports": 2,
    "Entertainment": 4
  },    
  "preferred_sources": ["source1", "source2"],
  "feedback_history": []
}
"""

# Load user data from JSON string
user = json.loads(user_data)

# Sample articles (titles and contents)
articles = [
    {"title": "Article 1", "content": "Content of article 1 about politics."},
    {"title": "Article 2", "content": "Content of article 2 about technology and business."},
    {"title": "Article 3", "content": "Content of article 3 about sports and entertainment."}
]

# Concatenate article titles and contents
corpus = [article["title"] + " " + article["content"] for article in articles]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(corpus)

# Calculate user profile vector
user_profile = np.zeros((1, tfidf_matrix.shape[1]))
for category, rating in user["user_preferences"].items():
    if category in vectorizer.vocabulary_:
        user_profile[0, vectorizer.vocabulary_[category.lower()]] = rating

# Calculate cosine similarity between user profile and articles
user_article_similarity = cosine_similarity(user_profile, tfidf_matrix)

# Get indices of top recommended articles
top_indices = user_article_similarity.argsort()[0][::-1]

# Recommend top articles
recommended_articles = [articles[i] for i in top_indices]

print("Recommended Articles:")
for article in recommended_articles:
    print(article["title"])
