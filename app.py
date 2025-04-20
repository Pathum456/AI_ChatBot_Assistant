import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
import random
import numpy as np
from preprocess import Preprocessor
from model import NeuralNet

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root", 
    password="root",  
    database="chatbot_db_test"
)
cursor = db.cursor()

# Ensure `chat_history` table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        message TEXT NOT NULL,
        response TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
db.commit()

# Load the trained model and data
FILE = "chatbot_model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Initialize preprocessor
preprocessor = Preprocessor()

# Load intents from JSON file
with open(r'D:\Esoft\AI\ChatBot\intents.json', 'r') as f:  # Use raw string or correct path
        intents = json.load(f)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get user message from request
        data = request.json
        user_message = data.get("message", "").strip()
        print("User Message : ",user_message)

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Generate bot response using the trained model
        tokenized_sentence = preprocessor.tokenize(user_message)
        print("tokenized : ",tokenized_sentence)
        X = preprocessor.bag_of_words(tokenized_sentence, all_words)
        print("x 1 : ",X)

        X = torch.from_numpy(X).to(device)
        print("x 2 : ",X)

        output = model(X.unsqueeze(0))
        _, predicted = torch.max(output, dim=1)
        print("predicted : ",predicted)
        print("_ : ",_)
        tag = tags[predicted.item()]
        print("tag : ",tag)

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print("prob : ",prob)
        if prob.item() > 0.75:
            for intent in intents['intents']:
                print("intent : ",intent)

                if intent['tag'] == tag:
                    bot_response = random.choice(intent['responses'])
                    break
        else:
            bot_response = "I'm sorry, I don't understand that."

        # Save chat history to the database
        cursor.execute("INSERT INTO chat_history (message, response) VALUES (%s, %s)", 
                       (user_message, bot_response))
        db.commit()

        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)