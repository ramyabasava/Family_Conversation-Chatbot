from flask import Flask, request, render_template_string
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import random

# Flask app
app = Flask(__name__)

# Load trained model & vectorizer (make sure you have saved them earlier)
loaded_vectorizer = joblib.load('tfidf_vectorizer.joblib')
loaded_model = joblib.load('naive_bayes_model.joblib')

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Responses
responses = {
    "greeting": ["Hello! How can I help you?", "Hi there!", "Good to see you."],
    "wellbeing": ["I hope you're doing well!", "Is everything okay?", "Feeling good, I hope!"],
    "food": ["What are you craving?", "Shall we order something?", "I'm thinking about dinner too."],
    "daily_conversation": ["Tell me about your day.", "That sounds interesting!", "What's next for you today?"],
    "farewell": ["Goodbye! Talk to you later.", "See you soon!", "Take care!"]
}

def generate_response(intent):
    if intent in responses:
        return random.choice(responses[intent])
    else:
        return "I'm not sure how to respond to that, but I'm learning!"

@app.route("/", methods=["GET", "POST"])
def home():
    user_message = ""
    chatbot_response = "Hi! I’m your Family Chatbot 😊 Type something below."
    if request.method == "POST":
        user_message = request.form["user_message"]
        if user_message.lower() == 'quit':
            chatbot_response = "Goodbye! 👋"
        else:
            cleaned_input = preprocess_text(user_message)
            if cleaned_input:
                input_vec = loaded_vectorizer.transform([cleaned_input])
                predicted_intent = loaded_model.predict(input_vec)[0]
                chatbot_response = generate_response(predicted_intent)
            else:
                chatbot_response = "Could you please rephrase that?"
    
    # HTML Template with CSS for neat design
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
        <title>Family Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; background: #f4f7f9; text-align: center; }
            .chat-container { width: 40%; margin: auto; background: #fff; padding: 20px; border-radius: 10px;
                              box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-top: 50px; }
            h1 { color: #333; }
            .chat-box { border: 1px solid #ddd; padding: 15px; border-radius: 5px; background: #fafafa; min-height: 100px; }
            .user { color: #0066cc; font-weight: bold; }
            .bot { color: #009933; font-weight: bold; }
            input[type="text"] { width: 70%; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
            input[type="submit"] { padding: 10px 15px; background: #0066cc; color: white; border: none; 
                                   border-radius: 5px; cursor: pointer; }
            input[type="submit"]:hover { background: #004c99; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <h1>👨‍👩‍👧 Family Chatbot</h1>
            <div class="chat-box">
                {% if user_message %}
                    <p><span class="user">You:</span> {{ user_message }}</p>
                {% endif %}
                <p><span class="bot">Chatbot:</span> {{ chatbot_response }}</p>
            </div>
            <form method="post">
                <input type="text" name="user_message" placeholder="Type your message..." autofocus required>
                <input type="submit" value="Send">
            </form>
        </div>
    </body>
    </html>
    """, user_message=user_message, chatbot_response=chatbot_response)

if __name__ == "__main__":
    app.run(debug=True)
