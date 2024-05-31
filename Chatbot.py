import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the FAQ data
faqs = {
    "What is SmartHome?": "SmartHome is a smart home automation system that allows you to control your home appliances remotely.",
    "How do I set up SmartHome?": "To set up SmartHome, download the app, create an account, and follow the in-app instructions to connect your devices.",
    "What devices are compatible with SmartHome?": "SmartHome is compatible with a wide range of devices, including lights, thermostats, security cameras, and more.",
    "Is SmartHome secure?": "Yes, SmartHome uses advanced encryption and secure servers to protect your data and ensure your home remains safe and secure.",
    "How much does SmartHome cost?": "SmartHome offers a free trial, and then it's $9.99/month or $99.99/year.",
}

# Define a function to preprocess the user's input
def preprocess_input(input_text):
    # Tokenize the input text
    tokens = word_tokenize(input_text)

    # Remove stopwords
    tokens = [token for token in tokens if token.lower() not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    input_text = ' '.join(tokens)

    return input_text

# Define a function to find the best matching FAQ
def find_best_match(input_text):
    best_match = None
    best_score = 0

    for question, answer in faqs.items():
        # Calculate the similarity between the input text and the FAQ question
        similarity = nltk.edit_distance(input_text, question) / len(question)

        # If the similarity is higher than the current best score, update the best match
        if similarity > best_score:
            best_match = answer
            best_score = similarity

    return best_match

# Define a function to respond to the user's input
def respond(input_text):
    input_text = preprocess_input(input_text)
    best_match = find_best_match(input_text)

    if best_match:
        return best_match
    else:
        return "Sorry, I didn't understand your question. Please try rephrasing it."

# Create a simple chatbot interface
while True:
    user_input = input("You: ")
    response = respond(user_input)
    print("SmartHome Assistant:", response)
