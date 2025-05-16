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
from datetime import datetime, timedelta
import json
import os
import subprocess

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

INTENTS_PATH = './assets/intents.json'

def export_intents_from_db():
    print(f"[{datetime.now()}] Starting scheduled export...")

    # Load existing intents
    if os.path.exists(INTENTS_PATH):
        with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
                intents = existing_data.get('intents', [])
            except json.JSONDecodeError:
                print("[ERROR] Couldn't parse existing intents.json — starting fresh.")
                intents = []
    else:
        intents = []

    intent_map = {intent['tag']: intent for intent in intents}
    newly_added_tags = []
    updated_tags = []
    skipped_rows = []

    # Today's date
    today_str = datetime.now().date().isoformat()

    # Fetch today's chat_history rows
    cursor.execute("SELECT * FROM chat_history WHERE DATE(timestamp) = CURDATE()")
    rows = cursor.fetchall()

    print(f"[INFO] Fetched {len(rows)} rows from chat_history for today's date:")

    for i, row in enumerate(rows, start=1):
        print(f"  [{i}] {row}")

    # Filter valid rows, skip if response is null/[]/empty
    valid_rows = []
    for row in rows:
        response = row.get('responses')
        tag = row.get('tag')
        pattern = row.get('pattern')

        if not response or response.strip() == '' or response.strip() == '[]':
            skipped_rows.append(row)
            # Update timestamp to tomorrow
            tomorrow = datetime.now() + timedelta(days=1)
            cursor.execute("UPDATE chat_history SET timestamp = %s WHERE id = %s", (tomorrow, row['id']))
            db.commit()
        else:
            valid_rows.append(row)

    # Build/merge intents
    for row in valid_rows:
        tag = row.get('tag')
        pattern = row.get('pattern')
        response = row.get('responses')
        context_set = row.get('context_set')

        if not tag or not pattern or not response:
            continue

        if tag not in intent_map:
            new_intent = {
                "tag": tag,
                "patterns": [pattern],
                "responses": [response],
                "context_set": context_set
            }
            intent_map[tag] = new_intent
            newly_added_tags.append(tag)
        else:
            # Update existing
            updated = False
            intent = intent_map[tag]

            if pattern not in intent['patterns']:
                intent['patterns'].append(pattern)
                updated = True

            if response not in intent['responses']:
                intent['responses'].append(response)
                updated = True

            if context_set and intent.get('context_set') != context_set:
                intent['context_set'] = context_set
                updated = True

            if updated and tag not in updated_tags:
                updated_tags.append(tag)

    # Final output
    final_data = {
        "intents": list(intent_map.values())
    }

    # Save to intents.json
    with open(INTENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)

    # Print summary
    if newly_added_tags:
        print(f"[INFO] New intents added: {', '.join(newly_added_tags)}")
    else:
        print("[INFO] No new intents added today.")

    if updated_tags:
        print(f"[INFO] Updated existing intents: {', '.join(updated_tags)}")
    else:
        print("[INFO] No existing intents were updated.")

    if skipped_rows:
        print(f"[INFO] Skipped {len(skipped_rows)} rows due to empty/null/[] responses and updated their timestamp to tomorrow.")
        for row in skipped_rows:
            print(f"  - Skipped ID {row['id']} (tag: {row['tag']}, response: {row['responses']})")

    print(f"[{datetime.now()}] Export complete. Total intents now: {len(final_data['intents'])}")

    # try:
    #     print("[INFO] Running training script (train.py)...")
    #     result = subprocess.run(['python', 'train.py'], check=True, capture_output=True, text=True)
    #     print("[INFO] Training completed successfully.")
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print(f"[ERROR] Training script failed:\n{e.stderr}")

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
    scheduler.add_job(export_intents_from_db, 'cron', hour=12, minute=32)  # Run daily at 11:30 AM
    scheduler.add_job(train_model, 'cron', hour=12, minute=33)  # Run daily at 11:30 AM
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

@app.route('/api/questions/<int:id>', methods=['DELETE'])
def delete_question(id):
    try:
        # Execute DELETE query
        cursor.execute("DELETE FROM chat_history WHERE id = %s", (id,))
        db.commit()

        # Return success response
        return jsonify({
            "status": "success",
            "message": f"Question with ID {id} deleted successfully."
        }), 200

    except Exception as e:
        db.rollback()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    
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
        # Check if question with this ID already exists in DB
        cursor.execute("SELECT id FROM chat_history WHERE id = %s", (item['id'],))
        result = cursor.fetchone()

        if result:
            # ID exists → Update
            sql = """
                UPDATE chat_history SET tag=%s, pattern=%s, responses=%s, context_set=%s
                WHERE id=%s
            """
            cursor.execute(sql, (
                item['tag'],
                item['pattern'],
                item['responses'],
                item.get('context_set'),
                item['id']
            ))
        else:
            # ID does not exist → Insert new row
            sql = """
                INSERT INTO chat_history (id, tag, pattern, responses, context_set)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                item['id'],
                item['tag'],
                item['pattern'],
                item['responses'],
                item.get('context_set')
            ))

    db.commit()
    return jsonify({"status": "success", "message": "Bulk upsert completed"})

# @app.route('/api/questions/<int:id>', methods=['DELETE'])
# def delete_question(id):
#     cursor.execute("DELETE FROM chat_history WHERE id = %s", (id,))
#     db.commit()
#     return jsonify({"status": "deleted"})

# Start scheduled tasks when app starts
if __name__ == '__main__':
    schedule_training()
    app.run(debug=False)