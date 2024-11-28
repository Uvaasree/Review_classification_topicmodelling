import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

# Ensure tokenizer and stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

class review_classify_cluster_topic_senti:
    
    # Initializer for the main class
    def __init__(self):
        self.lem = WordNetLemmatizer()
        self.stem = PorterStemmer()
    
    # Text extraction
    def text_extract(self, text):
        # Remove non-alphabetic characters
        return re.sub(r'[^a-zA-Z\s]', "", text)

    # Preprocess text (tokenization, lemmatization, and removing stopwords)
    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return [self.lem.lemmatize(word) for word in tokens if word not in stopwords.words("english")]

    # Pre-process function
    # Pre-process function
    def pre_process(self, texts):
        if isinstance(texts, list):  # If it's a list, convert to pandas Series
            texts = pd.Series(texts)
        texts = texts.apply(self.text_extract)  # Clean text
        texts = texts.apply(lambda x: " ".join(self.preprocess_text(x)))  # Tokenize, lemmatize, and join tokens
        return texts

    
    # Classification method
    def classify(self, text):
        text = self.pre_process([text])  # Preprocess the input text
        with open('class_log.pkl', 'rb') as f:
            model = pickle.load(f)
        return model.predict([text.iloc[0]])  # Predict the class
    
    # Clustering method
    def cluster(self, text):
        text = self.pre_process([text])  # Preprocess the input text
        with open('clustering_pipeline.pkl', 'rb') as f:
            model = pickle.load(f)
        return model.predict([text.iloc[0]])  # Predict the cluster
    
    # Topic modeling method using NMF
    def topic(self, text):
        def extract_topics_with_nmf(texts, num_topics=3, num_words=5):
            # Vectorize the texts
            vectorizer = CountVectorizer(stop_words='english')
            X = vectorizer.fit_transform(texts)

            # Apply NMF for topic modeling
            nmf_model = NMF(n_components=num_topics, random_state=42)
            W = nmf_model.fit_transform(X)  # Document-topic matrix
            H = nmf_model.components_  # Topic-term matrix

            # Extract top words for each topic
            topics = []
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(H):
                top_words_idx = topic.argsort()[-num_words:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(f"Topic {topic_idx + 1}: " + " ".join(top_words))

            return topics
        
        text = self.pre_process([text])  # Preprocess the input text
        topics = extract_topics_with_nmf(text, num_topics=3, num_words=5)
        return topics

    # Sentiment classification method
    def senti(self, text):
        text = self.pre_process([text])  # Preprocess the input text
        with open('sentiment_analysis.pkl', 'rb') as f:
            model = pickle.load(f)
            result = model([text.iloc[0]])  # Predict sentiment
            label = result[0]['label']
        return label
