from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Example dataset
data = {
    'Question': [
        'How are you?', 'Hello', "What's your name?", 'Tell me a joke', 'Goodbye', 
        'Hallo', 'Hey', 'What is the time?', 'Who is the president?', 'Tell me a fact', 
        'See you later', 'How old are you?', 'What can you do?', 'Do you have a hobby?', 
        'Thank you', 'Can you help me?', 'What’s your favorite color?', 
        'Tell me something interesting', 'Where are you from?', 'Do you speak other languages?', 
        'What’s your purpose?', 'Goodnight', 'What is your favorite movie?', 
        'Sing me a song', 'Who created you?', 'Nice to meet you', 'I need help', 
        'What do you think about AI?', 'Good morning', 'Can you tell me a story?'
    ],
    'Label': [
        'greeting', 'greeting', 'personal_info', 'entertainment', 'goodbye', 
        'greeting', 'greeting', 'information_request', 'information_request', 
        'entertainment', 'goodbye', 'personal_info', 'capability', 'personal_info', 
        'gratitude', 'assistance', 'personal_info', 'entertainment', 'personal_info', 
        'capability', 'purpose', 'goodbye', 'entertainment', 'entertainment', 
        'personal_info', 'greeting', 'assistance', 'opinion', 'greeting', 'entertainment'
    ]
}

# Train model
df = pd.DataFrame(data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Question'])
y = df['Label']
model = MultinomialNB()
model.fit(X, y)

# Route to handle requests
@app.route('/chat', methods=['POST','GET'])
def chat():
    user_input = request.json.get('message')  # Get the user's message
    user_input_vec = vectorizer.transform([user_input])  # Vectorize the input
    prediction = model.predict(user_input_vec)[0]  # Predict the label
    
    # Response based on predicted label
    if prediction == 'greeting':
        response = "Hello! How can I assist you today?"
    elif prediction == 'personal_info':
        response = "Im an AI chatbot, here to help with your questions!"
    elif prediction == 'entertainment':
        response = "Why dont scientists trust atoms? Because they make up everything!"
    elif prediction == 'goodbye':
        response = "Goodbye! Have a great day!"
    elif prediction == 'information_request':
        response = "I'm not sure about that, but I can look it up for you!"
    elif prediction == 'capability':
        response = "I can help answer questions, tell jokes, and provide information!"
    elif prediction == 'gratitude':
        response = "You're welcome! If you need anything else, just let me know."
    elif prediction == 'assistance':
        response = "Of course! How can I assist you today?"
    elif prediction == 'purpose':
        response = "I'm here to help answer your questions and provide information!"
    elif prediction == 'opinion':
        response = "That's an interesting question! I think AI has a lot of potential."
    else:
        response = "I'm sorry, I don't understand that."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
