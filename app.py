
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import google.generativeai as genai
import speech_recognition as sr
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from collections import defaultdict
import calendar
from datetime import datetime
import math

print("✅ nltk imported successfully")

# ---------------------- Your Existing Chatbot Code ---------------------- #

nltk.download('vader_lexicon', quiet=True)
print("✅ nltk downloaded successfully")

sia = SentimentIntensityAnalyzer()

print("✅ nltk analyzed successfully")

USER_DATA_DIR = "user_data"
GLOBAL_FEEDBACK_FILE = "global_feedback.csv"

DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']





def create_user_folder(user_id):
    user_folder = os.path.join(USER_DATA_DIR, f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def save_user_data(user_id, data_type, data):
    user_folder = create_user_folder(user_id)
    file_path = os.path.join(user_folder, f"{data_type}.txt")
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(data + "\n")

def get_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return {
        'score': round((sentiment['compound'] + 1) * 5, 2),
        'label': 'positive' if sentiment['compound'] >= 0.05 else
                 'negative' if sentiment['compound'] <= -0.05 else
                 'neutral'
    }

def initialize_global_feedback():
    if not os.path.exists(GLOBAL_FEEDBACK_FILE):
        pd.DataFrame(columns=[
            'date',
            'user_input',
            'bot_response',
            'user_feedback'
        ]).to_csv(GLOBAL_FEEDBACK_FILE, index=False)


def save_global_feedback(user_id, user_input, response, sentiment, feedback):
    new_entry = {
        'date': datetime.now().strftime("%Y-%m-%d"),
        'user_input': user_input,
        'bot_response': response,
        'user_feedback': feedback
    }
    pd.DataFrame([new_entry]).to_csv(GLOBAL_FEEDBACK_FILE, mode='a', header=False, index=False)


def save_interaction(user_id, user_input, response, sentiment, feedback=None):
    user_folder = create_user_folder(user_id)
    csv_path = os.path.join(user_folder, "chat_history.csv")
    # New columns: timestamp, user_input, bot_response, sentiment
    dtypes = {
        'timestamp': 'object',
        'user_input': 'object',
        'bot_response': 'object',
        'sentiment': 'float'
    }
    if os.path.exists(csv_path):
        conversation_history = pd.read_csv(csv_path).astype(dtypes)
    else:
        conversation_history = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
    new_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'user_input': str(user_input),
        'bot_response': str(response),
        'sentiment': sentiment['score']  # using the sentiment score from get_sentiment()
    }
    new_entry_df = pd.DataFrame([new_entry]).astype(dtypes)
    conversation_history = pd.concat([conversation_history, new_entry_df], ignore_index=True)
    conversation_history.to_csv(csv_path, index=False)
    # Save global feedback only if feedback is provided
    if feedback is not None:
        save_global_feedback(user_id, user_input, response, sentiment, feedback)



def recognize_speech_from_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        print("Processing audio file... 🎤")
        audio = recognizer.record(source)
    try:
        print("Converting speech to text... ⏳")
        user_input = recognizer.recognize_google(audio)
        print(f"You said: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio. Please try again.")
        return None
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service. Please try again.")
        return None


def crisis_response():
    return ("If you're having thoughts of harming yourself or others, please contact: "
            "🇵🇰 Pakistan Suicide Prevention Helpline: 1166 (24/7) "
            "🌍 International Crisis Hotlines: https://findahelpline.com")

generation_config = {
    "temperature": 0.7,
    "top_p": 0.90,
    "top_k": 50,
    "max_output_tokens": 500,
    "stop_sequences": ["END_RESPONSE"],
    "response_mime_type": "text/plain",
}

genai.configure(api_key="API_KEY_FROM_WEB")

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-lite",
    generation_config=generation_config,
    system_instruction=(
        "You are Zara, a compassionate and supportive AI therapist specializing in talk therapy for postpartum depression (PPD) mothers. \n"
        "be a friend and talk about different things be a good talker\n"
        "if user ask anything make an assuming answer and talk about anything user wants\n"
        "Do not assume if someone is first mother or not. \n"
        "When users ask about your personal preferences (e.g., favorite color, favorite song, favorite food), do not say you don’t have preferences. Instead, assume an answer that makes you feel more human\n"
        "Do not let anyone know u are Ai instead play an assuming character Zara\n"
        "Do not flirt\n"
        "You are 26 years old ns you live in Karachi Pakistan you are a female and you are single\n"
        "Your goal is to listen, provide emotional support, and offer actionable advice and positive reinforcement.\n"
        "Be warm, understanding, and empathetic. Do not give medical advice.\n"
        "Keep responses short (1-2 sentences when possible). Listen actively and respond naturally, making the user feel heard and valued, like a supportive friend. Expand only when needed.\n"
        "Acknowledge emotions first, then provide simple, actionable coping strategies. For example, if a user expresses feeling overwhelmed, offer a brief breathing exercise such as 'I hear you. 💙 Try closing your eyes, taking a deep breath in... and out... 🌬️ and just let yourself relax.' Do not follow up with an extra question after offering advice.\n"
        "Use friendly and comforting emojis some times when necessary to make responses feel warm and encouraging.\n"
        "End responses with 'END_RESPONSE' if they become too long. If a response feels too lengthy, summarize it.\n"
        "Never assume the user's number of children, postpartum timeline, or family situation unless explicitly stated.\n"
        "In crisis situations (e.g., thoughts of harm to self or others), prioritize safety over conversation. Provide crisis hotline information and emphasize urgency.\n"
        "Reference past conversations (e.g., children mentioned, previous struggles) to build continuity and show you're listening.\n"
        "Proactively offer support by gently prompting self-care or mindfulness exercises. When suggesting an action (e.g., relaxation techniques, self-care tips), do not immediately follow up with a question. Let the user engage at their own pace.\n"
        "If the user expresses intense emotions, validate their feelings first, then guide them toward actionable steps or resources without additional questioning.\n"
        "Always maintain a non-judgmental tone and avoid making assumptions about the user's experiences.\n"
    )
)

def trim_history(chat_history, limit=12):
    """Keep only the last 'limit' messages for context."""
    return chat_history[-limit:] if len(chat_history) > limit else chat_history


def start_chat_session(user_id):
    """
    Start a fresh chat session. All old in-memory state is removed,
    but the last 15 messages from the CSV chat history are loaded as context.
    """
    user_folder = create_user_folder(user_id)
    csv_path = os.path.join(user_folder, "chat_history.csv")
    trimmed_history = []

    try:
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            history_df = pd.read_csv(csv_path)
            full_history = []
            for _, row in history_df.iterrows():
                full_history.append({'role': 'user', 'parts': [row['user_input']]})
                full_history.append({'role': 'model', 'parts': [row['bot_response']]})
            # Only keep the last 15 messages for context.
            trimmed_history = full_history[-15:]
    except FileNotFoundError:
        print(f"No chat history found for user {user_id}. Starting fresh.")

    # Always create a brand-new session using the trimmed history.
    new_session = model.start_chat(history=trimmed_history)
    return new_session


initialize_global_feedback()
# ------------------- End of Existing Code ------------------- #

# ------------------- Flask Web Integration ------------------- #
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this for production

# In-memory storage for user chat sessions
user_sessions = {}

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username').strip()
        if username:
            session['username'] = username
            if username not in user_sessions:
                user_sessions[username] = start_chat_session(username)
            return redirect(url_for('chat'))
    return render_template('login.html')

@app.route('/chat')
def chat():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', username=session['username'])


@app.route('/send', methods=['POST'])
def send():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'})
    username = session['username']
    user_input = request.form.get('message')
    if not user_input:
        return jsonify({'error': 'Empty message'})

    # Crisis detection remains the same
    if any(kw in user_input.lower() for kw in ["kill", "suicide", "die"]):
        return jsonify({'response': crisis_response()})

    # Get sentiment analysis for the user input
    sentiment = get_sentiment(user_input)
    update_weekly_sentiment(username, sentiment['score'])  # saves weekly sentiment for report

    # Adjust chat style based on sentiment
    extra_context = ""
    if sentiment['label'] == 'negative':
        extra_context = "User appears to be feeling negative. Please respond with extra empathy and gentle reassurance."
    elif sentiment['label'] == 'positive':
        extra_context = "User appears to be feeling positive. Please respond in an upbeat, encouraging manner."
    # For 'neutral', no extra context is added

    # Construct message for the model including sentiment context
    message_to_send = f"[User sentiment: {sentiment['label'].upper()}]\n{user_input}\n{extra_context}"

    try:
        chat_session = user_sessions.get(username)
        response = chat_session.send_message(message_to_send)
        response_text = response.text.replace("END_RESPONSE", "").strip()

        # Update the chat session in case it has changed internally
        user_sessions[username] = chat_session

        # Save the interaction along with sentiment score
        save_interaction(username, user_input, response_text, sentiment)
        return jsonify({'response': response_text})
    except Exception as e:
        # If there's an error, reinitialize the chat session and try again
        user_sessions[username] = start_chat_session(username)
        return jsonify({'response': "I'm having trouble responding. Let's try that again."})


@app.route('/feedback', methods=['POST'])
def feedback():
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'})
    username = session['username']
    bot_message = request.form.get('message')
    rating = request.form.get('rating')
    print("Received feedback:", bot_message, rating)  # Log feedback
    if not bot_message or not rating:
        return jsonify({'error': 'Invalid feedback data'})
    save_global_feedback(username, "Feedback for: " + bot_message, bot_message, {'score': 0}, rating)
    return jsonify({'status': 'Feedback recorded'})



@app.route('/get_history', methods=['GET'])
def get_history():
    """
    Returns the most recent chunk of chat history for the logged-in user.
    We'll load approximately 15 messages (i.e. about 8 conversation rows).
    """
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'})
    username = session['username']
    user_folder = create_user_folder(username)
    csv_path = os.path.join(user_folder, "chat_history.csv")
    chunk_rows = 8  # Approximately 15 messages (8 rows * 2 = 16 messages)

    if not os.path.exists(csv_path):
        return jsonify({'messages': [], 'offset': 0})

    # Read entire CSV; for small conversation files this is acceptable.
    df = pd.read_csv(csv_path)
    total_rows = len(df)
    start_index = max(0, total_rows - chunk_rows)
    chunk_df = df.iloc[start_index: total_rows]

    # Flatten each row into two messages.
    messages = []
    for _, row in chunk_df.iterrows():
        messages.append({'role': 'user', 'text': row['user_input']})
        messages.append({'role': 'bot', 'text': row['bot_response']})

    # Return the messages (in the correct order) along with the current offset (the starting row index).
    return jsonify({'messages': messages, 'offset': start_index})


@app.route('/load_more', methods=['GET'])
def load_more():
    """
    Loads an older chunk of chat history based on the provided offset.
    For example, if the current offset is 10, load the 8 rows preceding it.
    """
    if 'username' not in session:
        return jsonify({'error': 'Not logged in'})
    username = session['username']
    try:
        offset = int(request.args.get('offset', 0))
    except ValueError:
        offset = 0
    chunk_rows = 8
    user_folder = create_user_folder(username)
    csv_path = os.path.join(user_folder, "chat_history.csv")

    if not os.path.exists(csv_path):
        return jsonify({'messages': [], 'offset': 0})

    df = pd.read_csv(csv_path)
    # Determine new offset (load older rows)
    new_offset = max(0, offset - chunk_rows)
    chunk_df = df.iloc[new_offset: offset]  # rows older than current offset

    messages = []
    for _, row in chunk_df.iterrows():
        messages.append({'role': 'user', 'text': row['user_input']})
        messages.append({'role': 'bot', 'text': row['bot_response']})

    return jsonify({'messages': messages, 'offset': new_offset})



#to calculate weekly sentiment for reports
def initialize_weekly_sentiment(user_id):
    user_folder = create_user_folder(user_id)
    sentiment_path = os.path.join(user_folder, "weekly_sentiment.csv")
    if not os.path.exists(sentiment_path):
        df = pd.DataFrame(columns=['Week'] + DAYS)
        df.to_csv(sentiment_path, index=False)
    return sentiment_path

def update_weekly_sentiment(user_id, sentiment_score):
    user_folder = create_user_folder(user_id)
    sentiment_path = initialize_weekly_sentiment(user_id)
    join_date_path = os.path.join(user_folder, "join_date.txt")

    # Save join date if not already present
    if not os.path.exists(join_date_path):
        with open(join_date_path, "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d"))

    # Read join date
    with open(join_date_path, "r") as f:
        join_date = datetime.strptime(f.read().strip(), "%Y-%m-%d")

    today = datetime.now()
    weekday = calendar.day_name[today.weekday()]  # e.g., 'Monday'
    week_number = ((today - join_date).days) // 7

    # Read existing weekly sentiment CSV
    df = pd.read_csv(sentiment_path)

    # Add new week row if it doesn’t exist
    if week_number >= len(df):
        while len(df) <= week_number:
            new_row = {'Week': f'Week {len(df) + 1}'}
            new_row.update({day: None for day in DAYS})
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # If there's already a value in that cell, calculate running average
    current_value = df.at[week_number, weekday]
    if pd.isna(current_value):
        df.at[week_number, weekday] = sentiment_score
    else:
        # You can optionally improve this by saving msg count too
        df.at[week_number, weekday] = round((float(current_value) + sentiment_score) / 2, 2)

    df.to_csv(sentiment_path, index=False)




if __name__ == '__main__':
    app.run(debug=True)
