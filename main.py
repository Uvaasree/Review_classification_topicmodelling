from class_def import review_classify_cluster_topic_senti 
from flask import Flask,jsonify,request
import numpy as np

app=Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Parse JSON data from request body
    text = data.get('text', '')  # Extract 'text' from the JSON

    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'Invalid input. Text must be a non-empty string'}), 400

    class_fun = review_classify_cluster_topic_senti()

    # Ensure text is passed as a single string
    label = class_fun.classify(text)
    cluster = class_fun.cluster(text)
    topic = class_fun.topic(text)
    sentiment = class_fun.senti(text)

    # Prepare output
    output = {
        'Classification': label,
        'Sentiment': sentiment,
        'Cluster': cluster,
        'Topics': topic
    }
    # Convert any ndarray to list to make it JSON serializable
    for key, value in output.items():
        if isinstance(value, np.ndarray):  # Check if the value is a NumPy ndarray
            output[key] = value.tolist()  # Convert ndarray to list
    
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)