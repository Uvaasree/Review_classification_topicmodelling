import streamlit as st
import pandas as pd
from class_def import review_classify_cluster_topic_senti

# Initialize the function from your class
class_fun = review_classify_cluster_topic_senti()

# Streamlit title and text input widget
st.title("Text Classification, Clustering, Topic Modeling & Sentiment Analysis")
st.write("Enter your text to get the analysis results:")

# Input field for the text
text = st.text_area("Input Text", "")

# Function to process the text and display the output
if text:
    # Ensure that text is passed as a single string (not a list)
    label = class_fun.classify(text)  # Pass text as a string
    cluster = class_fun.cluster(text)  # Pass text as a string
    topic = class_fun.topic(text)  # Pass text as a string
    sentiment = class_fun.senti(text)  # Pass text as a string

    # Prepare the output in a dictionary
    output = {
        'Classification': label,
        'Sentiment': sentiment,
        'Cluster': cluster,
        'Topics': topic
    }

    # Convert the output dictionary to a DataFrame and display it
    output_df = pd.DataFrame([output])
    st.write(output_df)

# streamlit run review_app.py