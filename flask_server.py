import firebase_admin
from firebase_admin import credentials
from flask import Flask, request, jsonify
from firebase_admin import firestore
import recommender as Recommender
import numpy as np
import pandas as pd

cred = credentials.Certificate("private/cs7is5-teamhera-firebase-adminsdk-sa9b4-ff4a9b855d.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

@app.route('/get_documents', methods=['GET'])
def get_documents():
    with app.app_context():
        db = firestore.client()
        documents = db.collection('documents').get()
        data = [doc.to_dict() for doc in documents]
        return jsonify(data)
    
@app.route('/get_users', methods=['GET'])
def get_users():
    with app.app_context():
        db = firestore.client()
        documents = db.collection('users').get()
        data = [doc.to_dict() for doc in documents]
        return jsonify(data)

@app.route('/get_personalized_news', methods=['GET'])
def personalized_news():
    print("Personalized news")
    user_id = request.args.get('userId')

    if not user_id:
        print("User id not provided")
        return jsonify({"error": "user_id is required"}), 400

    with app.app_context():
        print("Connecting to firestore")
        db = firestore.client()
        print("Connected to firestore")

        # Get user data
        user = db.collection('users').document(user_id).get()
        print("User data")
        if not user.exists:
            print("User not found")
            return jsonify({"error": "User not found"}), 404

        print("User found")
        # get user data
        user_data = user.to_dict()


        # print check if user_keywords is empty which is a type dict
        print("Checking if user_keywords is empty")
        if user_data['user_keywords'] == {}:
            print("User keywords is empty")
        # print if user_categories is empty
        print("Checking if user_categories is empty")
        if user_data['user_categories'] == {}:
            print("User categories is empty")

        # print if liked_articles is empty
        print("Checking if liked_articles is empty")
        if user_data['liked_articles'] == []:
            print("Liked articles is empty")

        # print if disliked_articles is empty
        print("Checking if disliked_articles is empty")
        if user_data['disliked_articles'] == []:
            print("Disliked articles is empty")
        # print if user_history is empty
        print("Checking if user_history is empty")
        if user_data['user_history'] == {}:
            print("User history is empty")


        # call recommender_news_based_on_keywords
        print("Getting personalised news")
        recommended_news = get_users_personalised_articles_based_on_keyword_and_preferences(user=user_data, data=get_documents().json)

        recommended_news = convert_to_serializable(recommended_news)

        # return the recommended news
        return jsonify(recommended_news)

def convert_to_serializable(obj):

    # if DataFrame
    if isinstance(obj, pd.DataFrame):
        obj.columns = obj.columns.str.lower()
        return obj.to_dict(orient='records')

    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    except Exception as e:
        print("Error converting to serializable format:", e)
        return None

def get_users_personalised_articles_based_on_keyword_and_preferences(user, data):
    # load data to recommender and user
    recommender.load_data(data)
    recommender.load_user(user)

    # Call the method for recommending news
    print("Returning recommended news")
    return recommender.recommend_news_based_on_keyword_and_preferences()

# Add data to database
@app.route('/add_data', methods=['POST'])
def add_data():
    with app.app_context():
        db = firestore.client()
        data = request.json
        db.collection('documents').add(data)
        return jsonify({"message": "success"})





if __name__ == '__main__':
    print("Creating recommender object")
    recommender = Recommender.Recommender()

    print("Loading glove model")
    recommender.load_glove_model("gloves/converted_vectors300.txt")
    print("Loaded glove model")

    print("Starting flask server")
    app.run(debug=True, host='localhost', port=5000)










# user data that is stored in the firestore databse:

"""

{
    "user_id": "123456789",
    "username": "example_user",
    "email": "user@example.com",
    "password_hash": "hashed_password_here",
    "profile_picture": "https://example.com/profile_picture.jpg",
    "user_history": {
        "articles": [
            {
                "title": "Study Reveals Surprising Link Between Coffee Consumption and Memory Enhancement",
                "category": "Life"
            }
        ]
    },
    "user_keywords": {
        "advancements": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "artificial": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "intelligence": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "shaping": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "future": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "technology": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "exploring": {
            "score": 0.8,
            "last_modified": 1712050144.041242
        },
        "philosophy": {
            "score": 0.8,
            "last_modified": 1712050144.041242
        },
        "existentialism": {
            "score": 0.8,
            "last_modified": 1712050144.041242
        },
        "human": {
            "score": 0.18000000000000002,
            "last_modified": 1712050062.390597
        },
        "understanding": {
            "score": 2.5920000000000005,
            "last_modified": 1712050062.390597
        },
        "psychology": {
            "score": 0.8,
            "last_modified": 1712050062.390597
        },
        "decision": {
            "score": 0.8,
            "last_modified": 1712050062.390597
        },
        "making": {
            "score": 0.8,
            "last_modified": 1712050062.390597
        }
    },
    "user_categories": {
        "Science": {
            "score": 0.8,
            "last_modified": 1712051013.130931
        },
        "OnWeb": {
            "score": 2.5920000000000005,
            "last_modified": 1712050144.041242
        },
        "Art": {
            "score": 1.4400000000000002,
            "last_modified": 1712050058.081056
        },
        "Environment": {
            "score": 1.4400000000000002,
            "last_modified": 1712050036.192425
        },
        "Business": {
            "score": 0.0,
            "last_modified": 1712049945.281864
        },
        "Life": {
            "score": 0.1,
            "last_modified": 1712048431.113002
        },
        "Technology": {
            "score": 0,
            "last_modified": 1712048431.113002
        },
        "Nature": {
            "score": 0.8,
            "last_modified": 1711820408.861159
        },
        "Self-Help": {
            "score": 0.8,
            "last_modified": 1711820402.950997
        }
    },
    "liked_articles": [
        {
            "title": "Healthy Eating Habits: Tips for a Balanced Diet and Nutritional Wellness",
            "category": "Health"
        },
        {
            "title": "The Importance of Exercise: Enhancing Physical Health and Fitness",
            "category": "Health"
        },
        {
            "title": "Exploring the Philosophy of Existentialism: Finding Meaning in a Chaotic World",
            "category": "Education"
        },
        {
            "title": "Advancements in Artificial Intelligence: Shaping the Future of Technology",
            "category": "Science"
        }
    ],
    "disliked_articles": []
}

"""
