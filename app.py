import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import mysql.connector
import torch
import random
import numpy as np
from preprocess import Preprocessor
from model import NeuralNet
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import atexit

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Connect to MySQL database
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",
    database="chatbot_db_test"
)
cursor = db.cursor(dictionary=True)
db.commit()

# Ensure chat_history table exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INT AUTO_INCREMENT PRIMARY KEY,
        tag VARCHAR(255) NOT NULL,
        pattern TEXT NOT NULL,
        responses TEXT NOT NULL,
        context_set VARCHAR(255),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
db.commit()

# Load trained model and data
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

# Load intents
with open(r'D:\Esoft\AI\ChatBot\intents.json', 'r') as f:
    intents = json.load(f)


@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        print("User Message :", user_message)

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        tokenized_sentence = preprocessor.tokenize(user_message)
        X = preprocessor.bag_of_words(tokenized_sentence, all_words)
        X = torch.from_numpy(X).to(device)

        output = model(X.unsqueeze(0))
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        bot_response = "I'm sorry, I don't understand that."
        matched_intent = None

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if intent['tag'] == tag:
                    bot_response = random.choice(intent['responses'])
                    matched_intent = intent
                    break  # Stop after match

        # Only log to chat_history if no intent was matched
        if bot_response == "I'm sorry, I don't understand that.":
            cursor.execute("""
                INSERT INTO chat_history (tag, pattern, responses, context_set)
                VALUES (%s, %s, %s, %s)
            """, (
                'unknown',
                user_message,
                json.dumps([]),  # No valid response
                None             # Context not used
            ))
            db.commit()

        return jsonify({"response": bot_response})

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500

# --- Schedule training task ---
def train_model():
    print("Running train.py...")
    try:
        result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
        print("Training completed.")
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except Exception as e:
        print("Error during training:", str(e))


def schedule_training():
    print("Scheduling training task...")
    scheduler = BackgroundScheduler()
    scheduler.add_job(train_model, 'cron', hour=12, minute=30)  # Run daily at 12:30 AM
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())

@app.route('/api/questions', methods=['GET'])
def get_questions():
    cursor.execute("SELECT * FROM chat_history ORDER BY id ASC")
    return jsonify(cursor.fetchall())

@app.route('/api/questions/<int:id>', methods=['GET'])
def get_question(id):
    cursor.execute("SELECT * FROM chat_history WHERE id = %s", (id,))
    result = cursor.fetchone()
    return jsonify(result) if result else ('', 404)

@app.route('/api/questions', methods=['POST'])
def add_or_update_question():
    data = request.get_json()
    if 'id' in data:
        # Update
        sql = """
        UPDATE chat_history SET tag=%s, pattern=%s, responses=%s, context_set=%s
        WHERE id=%s
        """
        cursor.execute(sql, (
            data['tag'], data['pattern'], data['response'],
            data.get('context'), data['id']
        ))
    else:
        # Insert
        sql = """
        INSERT INTO chat_history (tag, pattern, responses, context_set)
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(sql, (
            data['tag'], data['pattern'], data['response'], data.get('context')
        ))
    db.commit()
    return jsonify({"status": "success"})

@app.route('/api/questions/bulk', methods=['POST'])
def bulk_update_questions():
    data = request.get_json()
    for item in data:
        sql = """
        UPDATE chat_history SET tag=%s, pattern=%s, responses=%s, context_set=%s
        WHERE id=%s
        """
        cursor.execute(sql, (
            item['tag'], item['pattern'], item['responses'],
            item.get('context_set'), item['id']
        ))
    db.commit()
    return jsonify({"status": "bulk updated"})

@app.route('/api/questions/<int:id>', methods=['DELETE'])
def delete_question(id):
    cursor.execute("DELETE FROM chat_history WHERE id = %s", (id,))
    db.commit()
    return jsonify({"status": "deleted"})

# Start scheduled tasks when app starts
if __name__ == '__main__':
    schedule_training()
    app.run(debug=False)