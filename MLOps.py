from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)  # Corrected: use __name__

# Load the pre-trained sarcasm detection model
model = tf.keras.models.load_model('/app/models/my_model.h5')

# Load the tokenizer from the tokenizer.pickle file
with open('/app/models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Pre-processing function to tokenize and pad the input text
def preprocess_text(text, max_len=100):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence to ensure it matches the input shape required by the model
    padded_sequence = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    
    return np.array(padded_sequence)

@app.route('/')
def home():
    return "Welcome to Sarcasm Detection API!"

# Define the /predict API route
@app.route('/predict', methods=['POST'])
def predict_sarcasm():
    try:
        # Extract JSON data from POST request
        data = request.get_json(force=True)
        text = data.get('text', None)

        if not text:
            return jsonify({'error': 'Missing "text" field in request body'}), 400

        # Pre-process the input text (tokenizing and padding)
        processed_text = preprocess_text(text)

        # Make a prediction using the pre-trained model
        prediction = model.predict(processed_text)

        # Assuming binary classification, set a threshold for sarcasm detection
        sarcasm_detected = bool(prediction[0] > 0.5)

        # Return the result as a JSON response
        return jsonify({'sarcasm': sarcasm_detected})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':  # Corrected: use __name__ and __main__
    app.run(host='0.0.0.0', port=5000, debug=True)
