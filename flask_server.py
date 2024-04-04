import firebase_admin
from firebase_admin import credentials
from flask import Flask, request, jsonify
from firebase_admin import firestore

cred = credentials.Certificate("private/cs7is5-teamhera-firebase-adminsdk-sa9b4-8a9cc23951.json")
firebase_admin.initialize_app(cred)

app = Flask(__name__)

@app.route('/get_data', methods=['GET'])
def get_data():
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




# Add data to database
@app.route('/add_data', methods=['POST'])
def add_data():
    with app.app_context():
        db = firestore.client()
        data = request.json
        db.collection('documents').add(data)
        return jsonify({"message": "success"})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
    print(get_data())
