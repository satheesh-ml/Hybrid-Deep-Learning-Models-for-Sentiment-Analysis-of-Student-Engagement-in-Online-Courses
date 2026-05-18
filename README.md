# Hybrid-Deep-Learning-Models-for-Sentiment-Analysis-of-Student-Engagement-in-Online-Courses


Step 1: Data Collection
•	Dataset: 100K Coursera's Course Reviews Dataset
•	https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset?select=reviews.csv


Step 2: Data Preprocessing
 Text Cleaning
•	Removal of punctuation
•	Removal of special characters
•	Conversion of all text to lowercase
•	Handling of missing/null values
Tokenization
•	Conversion of sentences into word sequences (tokens)
 Sequence Padding
•	All sequences were padded to a fixed length to ensure uniform input size for the neural network
Train–Test Split
•	The dataset was divided into training and testing sets using a 70:30 ratio


 Step 3: Feature Extraction
Embedding Layer (Trainable Word Embedding)
Embedding dimension: 100 or 128
Input length: Based on max padding length 


Step 4: Hybrid Deep Learning Classification
Hybrid CNN + BiLSTM with Softmax multi-class classification.
•	CNN → Extracts local features
•	BiLSTM → Captures contextual dependencies
•	Dense Layer + Softmax → Final sentiment classification
Output:
•	Positive
•	Neutral
•	Negative


 Step 5: Optimization
•	Optimizer: Adam
•	Loss Function: Categorical Crossentropy
•	Hyperparameter tuning (epochs, batch size, learning rate)
•	Dropout to reduce overfitting


 Step 6: Evaluation
Model performance evaluated using:
•	Accuracy
•	Precision
•	Recall
•	F1-Score
•	Confusion Matrix
Comparison done with baseline models:
•	LSTM
•	CNN
•	Traditional ML (optional)

