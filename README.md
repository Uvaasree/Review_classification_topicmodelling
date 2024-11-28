# Review Classification and Topic Modelling

## Problem Statement

In the age of online commerce, reviews significantly influence consumer decisions. However, the prevalence of fake or misleading reviews distorts consumer perceptions and undermines trust.  

The goal of this project is to develop a system capable of:  
1. **Classifying reviews as fake or real** using traditional machine learning models, deep learning models, and Transformers.  
2. **Clustering similar reviews** to group related feedback.  
3. **Identifying underlying topics** in reviews to understand customer sentiments and issues.

## Approach

### 1. **Data Collection**
The project uses a Fake Review Dataset containing product reviews with textual content and metadata, such as ratings and helpful votes.

### 2. **Preprocessing**
- **Text Cleaning**: Remove special characters, numbers, and stopwords.  
- **Tokenization**: Split text into words or tokens.  
- **Vectorization**: Convert text to numerical features using methods like TF-IDF and Word2Vec.  

### 3. **Topic Modeling**  
Topic modeling is employed to identify key themes in reviews using:  
- **Latent Dirichlet Allocation (LDA)**: Assigns words to topics probabilistically.  
- **Non-Negative Matrix Factorization (NMF)**: Factorizes word occurrences to extract topics.  

### 4. **Clustering (Unsupervised Learning)**  
Groups similar reviews for pattern discovery:  
- **K-Means Clustering**: Groups reviews into K clusters.  
- **DBSCAN**: Identifies clusters without requiring predefined counts.  

### 5. **Fake Review Classification**  
Classifies reviews as fake or real using:  
- **Traditional Machine Learning**: Logistic Regression, Random Forest, and SVM.  
- **Deep Learning**: LSTM for sequential modeling and BERT for context-aware classification.

### 6. **Sentiment Analysis**  
Classifies sentiment using:  
- **Pretrained model**: pretrained model in hugging face.  

## Business Use Cases

- **Customer Trust**: Detect fake reviews to protect consumers from misleading information.  
- **Product Feedback Analysis**: Group similar reviews to highlight common issues or features.  
- **Content Moderation**: Automatically filter out fake or harmful reviews from product pages.  

---

## Implementation Details

### **Text Vectorization**
- Convert review text into numerical format using **TF-IDF,word2vector,Count Vectorizer**.  

### **Topic Modeling**
- Use **LDA** or **NMF** to extract topics across all reviews.  
- Assign topics based on word distributions.

### **Clustering**
- Apply **K-Means** or **DBSCAN** to group reviews.

 ### **Sentiment Analysis**
- Apply Pretrained model from hugging face to predict sentiment of a review.

### **Fake Review Classification**
1. Tokenize reviews.  
2. Train models like Logistic Regression, Random Forest, LSTM, and BERT.  
3. Evaluate using metrics like accuracy, precision, recall, and F1 score.

---

## Technical Stack

- **Programming Language**: Python  
- **Libraries**: NLTK, Scikit-learn, TensorFlow/PyTorch, Transformers, Gensimn  

---

## Results

The project evaluates models using:  
- **Classification Metrics**: Accuracy, Precision, Recall, and F1 Score.

## User Testing

The project evaluates models using:  
- **Streamlit Application**: Interactive streamlit application for user use.

---

