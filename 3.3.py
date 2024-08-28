#!/usr/bin/env python
# coding: utf-8

# In[194]:


get_ipython().system(' pip install pipreqs')


# In[ ]:


get_ipython().system(' pipreqs')


# # 0. Import Libraries 

# In[193]:


# For preprocessing
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import emoji
from urllib.parse import urlparse
import seaborn as sns
import matplotlib.pyplot as plt

from ClassificationScores import ClassificationEvaluator

# For VaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For RNN model 
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# For DistilBERT
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


# # 1. Data Preprocessing

# ### 1.1. Importing the data and getting known to it.

# In[2]:


# Importing the train and test datasets
train_df = pd.read_parquet('btc_tweets_train.parquet.gzip')
test_df = pd.read_parquet('btc_tweets_test.parquet.gzip')


# In[3]:


train_df = train_df.reset_index()
test_df = test_df.reset_index()


# In[4]:


def df_info(df):
    return df.shape, df.isnull().sum().sum(), df.info()


# In[5]:


# Running a few checks to understand the datasets
df_info(train_df)


# In[6]:


df_info(test_df)


# In[7]:


# Dropping the unnecessary features (Hashtags in the tweets are seperately handled ahead)
train_df = train_df.drop(['tweet ID', 'user_displayname', 'hashtags'], axis=1)
test_df = test_df.drop(['tweet ID', 'user_displayname', 'hashtags'], axis=1)


# In[8]:


# Converting the sentiment labels from bool to int
train_df['sentiment'] = train_df['sentiment'].astype(int)
test_df['sentiment'] = test_df['sentiment'].astype(int)


# ### 1.2. DataPreprocessor Class

# In[139]:





# In[188]:


# Initialize the class for train and test df
train_datapreprocessor = DataPreprocessor(df=train_df, content_column='content')
test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# ### 1.3. ClassificationEvaluator Class

# In[159]:





# ### A few trials

# In[ ]:


def extract_special_characters(text):
    # Regular expression to match special characters
    return re.findall(r'[^a-zA-Z0-9\s]', text)

# Extract special characters from the selected column (e.g., 'content')
df = train_df.copy()
df['special_chars'] = df['content'].apply(extract_special_characters)

# Flatten the list of special characters and count their frequency
special_chars_list = df['special_chars'].sum()
special_chars_count = Counter(special_chars_list)

# Get the most common special characters
most_common_special_chars = special_chars_count.most_common()

# Display the results
for char, count in most_common_special_chars:
    print(f"'{char}': {count}")


# In[ ]:



# In[60]:


def extract_emojis(text):
    return ''.join(char for char in text if char in emoji.EMOJI_DATA)

# Extract emojis from the selected column (e.g., 'content')
#column_to_use = 'cleaned_content' if 'cleaned_content' in df.columns else 'content'
df = train_df.copy()
df['emojis'] = df['content'].apply(extract_emojis)

# Flatten the list of emojis and count the frequency of each emoji
emoji_list = df['emojis'].sum()
emoji_count = Counter(emoji_list)

# Get the top 50 most used emojis
top_50_emojis = emoji_count.most_common(200)

# Display the results
for emoji_char, count in top_50_emojis:
    print(f"{emoji_char}: {count}")


# **NOTE: both the cleaned datasets above are still imbalanced with True values largly outnumbered than False. The imbalance needs to be taken care of by assigning class weights dring model training.**

# # 2. Benchmark: vaderSentiment Sentiment Dictionary

# For sentiment analysis using VADER, it's best to apply VADER to the raw, uncleaned text to leverage its strengths in handling informal language, punctuation, and emojis. However, when it comes to Links, it is best to remove them. Links are irrelevent to the sentiment and could add unnecessary noise, potentially influencing the sentiment.

# The preprocessing flow for vaderSentiment
# - Remove spam tweets
# - Remove links
# - Remove unnecessary hashtags
# - Then apply vader sentiment dictionary on the cleaned tweets

# ### 2.1. Getting the dataset ready for VaderSentiment

# In[24]:


vader_test_df = test_df.copy()
vader_datapreprocessor = DataPreprocessor(df=vader_test_df, content_column='content')


# In[25]:


# Getting the dataset ready by removing spam, unnecessary hashtags, links, whitespace and HTML
vader_test_df = vader_datapreprocessor.preprocess( remove_spam=True,
                                                   remove_hashtags=True, 
                                                   remove_link=True, 
                                                   remove_whitespace_html=True, 
                                                   remove_emoji=True)


# In[26]:


vader_test_df.head(20)


# In[27]:


# look over on example to see if the tweets are cleaned as expected
index = 4
vader_test_df['content'].iloc[index], vader_test_df['cleaned_content'].iloc[index]


# ### 2.2. Fitting the VaderSentiment on processed test data

# In[28]:


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to apply VADER sentiment analysis and get the compound score
def get_sentiment_score(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']  # 'compound' score is a normalized score between -1 (negative) and +1 (positive)

# Apply sentiment analysis to the 'content' column and create a new column for the sentiment score
vader_test_df['vader_sentiment'] = vader_test_df['cleaned_content'].apply(get_sentiment_score)

# Classify the sentiment based on the compound score
def classify_sentiment(score):
    if score >= 0:
        return True
    else:
        return False

# Apply the classification and create a new column for the sentiment label
vader_test_df['vader_sentiment_label'] = vader_test_df['vader_sentiment'].apply(classify_sentiment).astype(int)


# ### 2.3. Evaluating the vaderSentiment classification performance

# In[31]:


# Initializing the performance evaluator class for vader
vader_performance_evaluator = ClassificationEvaluator(true_labels=vader_test_df['sentiment'], predicted_labels=vader_test_df['vader_sentiment_label'])


# In[32]:


vader_performance_evaluator.evaluate()
vader_performance_evaluator.plot_confusion_matrix()


# # 3. RNN 

# ### 3.1. Getting the dataset ready for generating the embeddings

# In[141]:


# Remove spam samples
rnn_train_df = train_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_link=True, remove_whitespace_html=True, clean_text=True, balance_classes=False)
rnn_test_df = test_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_link=True, remove_whitespace_html=True, clean_text=True, balance_classes=False)


# In[142]:


i = 75
rnn_train_df['content'][i], rnn_train_df['cleaned_content'][i], rnn_train_df['sentiment'][i]


# ### 3.2. Training the FastText embeddings

# In[124]:


nltk.download('punkt')
nltk.download('wordnet')


# In[143]:


# Tokenize the cleaned tweets (split by spaces)
tokenized_tweets = [tweet.split() for tweet in rnn_train_df['cleaned_content']]

# Train FastText model using Gensim's implementation
fasttext_model = FastText(sentences=tokenized_tweets, vector_size=50, window=4, min_count=1, sg=1, epochs=6)

# Save the model
#fasttext_model.save("fasttext_vs50.model")

# Load the model (for future use)
#fasttext_model = FastText.load("fasttext_vs50.model")


# In[ ]:


# Example: Get vector for a word 
# print(f"Vector for 'bitcoin': {fasttext_model.wv['bitcoin']}")

# Example: Get most similar words
#print(f"Words similar to 'bitcoin': {fasttext_model.wv.most_similar('bitcoin')}")


# ### 3.3. Building and Training the RNN model

# In[144]:


# Parameters
max_sequence_length = 40
embedding_dim = 50
embedding_matrix = fasttext_model.wv.vectors # Create embedding matrix
vocab_size = len(fasttext_model.wv) #len(tokenizer.word_index)+1


# In[ ]:


#fasttext_model.wv.vector_size, len(tokenizer.word_counts), vocab_size, tokenizer.word_index.items()


# In[145]:


# Tokenize the tweets
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(rnn_train_df['cleaned_content'])

train_sequences = tokenizer.texts_to_sequences(rnn_train_df['cleaned_content']) # sequences are arrays where the each word is replaced by number which corresponds to the position of that word in the vocabulary
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(rnn_test_df['cleaned_content'])
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')


# In[ ]:


# Dimension check
padded_train_sequences.shape, rnn_train_df['sentiment'].shape


# In[146]:


# Split the data to training and validation
X_train, X_val, y_train, y_val = train_test_split(padded_train_sequences, rnn_train_df['sentiment'], test_size = 0.2, random_state = 9)
# Changing the dtype from series to array
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Preparing the test data
rnn_X_test = padded_test_sequences
rnn_y_test = rnn_test_df['sentiment'].to_numpy()


# In[129]:


# Calculate the class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
class_weights


# Study about kernel_regularizer, learning_rate

# In[147]:


# Build the RNN model
def build_rnn_model(embd_dim, rnn_type='LSTM'):
    model = Sequential()
    model.add(Input(shape=(max_sequence_length,)))
    model.add(Embedding(input_dim = vocab_size, 
                        output_dim = embd_dim, 
                        weights=[embedding_matrix],  
                        trainable=False))
    
    if rnn_type == 'LSTM':
        model.add(LSTM(units = 64, return_sequences = False))
    elif rnn_type == 'GRU':
        model.add(GRU(units = 64, return_sequences= False))
        
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=l2(0.02)))
    
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[148]:


# Initialize the model
model = build_rnn_model(embd_dim=embedding_dim, rnn_type='LSTM')
model.summary()


# In[149]:


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])


# In[150]:


# Evaluate the model
loss, accuracy = model.evaluate(rnn_X_test, rnn_y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


# In[154]:


# Predicting sentiments on new tweets
predictions = model.predict(rnn_X_test)
# Convert probabilities to binary predictions
predictions_discrete = (predictions > 0.5).astype(int)  


# In[160]:


# Initializing the performance evaluator class for RNN
RNN_performance_evaluator = PerformanceEvaluator(true_labels=rnn_y_test, predicted_labels=predictions_discrete, predicted_probs=predictions)


# In[161]:


RNN_performance_evaluator.evaluate()
RNN_performance_evaluator.plot_confusion_matrix()
RNN_performance_evaluator.plot_roc_auc()


# In[153]:


print(classification_report(rnn_y_test, predictions.flatten()))
confusion_matrix(rnn_y_test, predictions.flatten())


# # 4. Pre-trained DistilBERT

# ### Fine tuning DistilBERT

# In[174]:


from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, TextClassificationPipeline


# In[ ]:





# In[162]:


from transformers import pipeline


# In[163]:


classifier = pipeline('sentiment-analysis',
                      model = 'distilbert-base-uncased-finetuned-sst-2-english',
                      tokenizer='distilbert-base-uncased',
                      batch_size=16)


# In[165]:


classifier('Bitcoin in 2 months. Be ready for upcoming rocket ralley.')


# In[168]:


def pipeline_classify(data):
    ''' Function to run the sentiment analysis pipeline on each row of a dataset
    and extract the scores. '''

    predictions = []

    for row in data:
          classification = classifier(row, truncation=True)[0]
          if classification['label'] == 'POSITIVE':
              predictions.append(classification['score'])
          else:
              predictions.append(1-classification['score'])
              
    return predictions


# In[189]:


bert_test_df = test_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_link=True, remove_whitespace_html=True, remove_emoji=False)


# In[190]:


bert_test_df.head()


# In[191]:


bert_pred = pipeline_classify(bert_test_df['cleaned_content'])


# In[172]:


from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

def assess_sentiment_classifier(ytest, yhat, cutoff=0.5, plot_roc=True):
    ''' Function to assess a classification model in terms of the AUC
    and classification accuracy. '''

    # Calculate discrete class predictions
    yhat_discrete = np.where(pd.DataFrame(yhat)>cutoff, 1, 0)
    # Calculate classification accuracy and AUC
    acc = accuracy_score(ytest, yhat_discrete)
    auc = roc_auc_score(ytest, yhat)
    # Confusion matrix 
    cmat = confusion_matrix(ytest, yhat_discrete)
    # ROC analysis
    if plot_roc==True:
        fpr, tpr, _ = roc_curve(ytest, yhat)
        plt.plot(fpr,tpr, label="AUC={:.4}".format(auc));
        plt.plot([0, 1], [0, 1], "r--")
        plt.ylabel('True positive rate')    
        plt.xlabel('False positive rate')    
        plt.legend(loc='lower right')
        plt.show();
    
    print("NN test set performance:\tAUC={:.4f}\tAccuracy={:.4f}".format(auc, acc))
    print('Confusion matrix:')
    print(cmat)


# In[192]:


assess_sentiment_classifier(test_labels, bert_pred)


# In[ ]:


bert_pred


# In[ ]:


bert_train_df = train_datapreprocessor.preprocess(remove_spam=False, remove_hashtags=True, remove_link=True, remove_whitespace_html=True)
bert_test_df = test_datapreprocessor.preprocess(remove_spam=False, remove_hashtags=True, remove_link=True, remove_whitespace_html=True)


# In[ ]:


bert_train_df.head()


# In[ ]:


# Dataset Preparation for Handling Tweet Data
class TweetDataset:
    def __init__(self, texts, labels=None, tokenizer=None, max_len=40):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    # Tokenize the text samples and prepares them for input into the model.
    def encode(self):
        encodings = self.tokenizer(
            self.texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='tf'
        )
        return encodings
    
    # Converts the tokenized texts and labels into a TensorFlow Dataset object.
    def prepare_data(self):
        encodings = self.encode()
        if self.labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), self.labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(encodings))
        return dataset


# In[ ]:


# Class to Fit Pretrained DistilBERT and Fine-Tune It
class SentimentAnalyzer:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def fit_pretrained(self, test_texts, batch_size=16):
        test_dataset = TweetDataset(texts=test_texts, tokenizer=self.tokenizer).prepare_data()
        test_dataset = test_dataset.batch(batch_size)

        predictions = self.model.predict(test_dataset).logits
        predictions = tf.argmax(predictions, axis=-1).numpy()

        return predictions

    def fine_tune(self, train_texts, train_labels, test_texts, test_labels, batch_size=16, epochs=3):
        train_dataset = TweetDataset(texts=train_texts, labels=train_labels, tokenizer=self.tokenizer).prepare_data()
        test_dataset = TweetDataset(texts=test_texts, labels=test_labels, tokenizer=self.tokenizer).prepare_data()

        train_dataset = train_dataset.shuffle(len(train_texts)).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)
        
        # Assuming train_labels is your array of labels
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
        class_weight_dict = dict(enumerate(class_weights))

        #optimizer = Adam(learning_rate=5e-5)
        self.model.compile(optimizer='adam', 
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #self.model.compute_loss, 
                           metrics=['accuracy'])

        history = self.model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict
        )

        return history


# In[171]:


#train_texts = bert_train_df['cleaned_content'].tolist()
#train_labels = bert_train_df['sentiment'].tolist()
test_texts = bert_test_df['cleaned_content'].tolist()
test_labels = bert_test_df['sentiment'].tolist()


# In[ ]:


train_texts


# In[ ]:


# Initialize the SentimentAnalyzer
sentiment_analyzer = SentimentAnalyzer()


# ### Pretrained predictions
# 

# In[ ]:


# Fit Pretrained DistilBERT on Test Dataset
pretrained_predictions = sentiment_analyzer.fit_pretrained(test_texts=test_texts)


# In[ ]:


# Class to Evaluate the Model's Performance
class PerformanceEvaluator:
    def __init__(self):
        pass

    def evaluate(self, true_labels, predicted_labels):
        print('Classification report:\n')
        return print(classification_report(true_labels, predicted_labels))

    def plot_confusion_matrix(self, true_labels, predicted_labels):
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(6, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative (0)', 'Positive(1)'], yticklabels=['Negative (0)', 'Positive(1)'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()


# In[ ]:


# Initialize PerformanceEvaluator
evaluator = PerformanceEvaluator(true_labels=test_labels, predicted_labels=pretrained_predictions)
evaluator.evaluate()
evaluator.plot_confusion_matrix()


# In[ ]:


np.unique(test_labels, return_counts=True)


# In[ ]:


np.unique(pretrained_predictions, return_counts=True)


# ### Fine tuned predictions

# In[ ]:


# Fine tune on train dataset and apply on test dataset
sentiment_analyzer.fine_tune(train_texts, train_labels, test_texts, test_labels)


# In[ ]:


fine_tuned_predictions = sentiment_analyzer.fit_pretrained(test_texts)


# In[ ]:


fine_tuned_predictions


# In[ ]:


evaluator.evaluate(true_labels=test_labels, predicted_labels=fine_tuned_predictions)
evaluator.plot_confusion_matrix(true_labels=test_labels, predicted_labels=fine_tuned_predictions)


# In[ ]:


print(tf.__version__)


# In[ ]:


import transformers
print(transformers.__version__)


# In[ ]:





# In[ ]:


classifier = pipeline('sentiment-analysis')


# In[ ]:


res = classifier(cleaned_train_df['content'][3])


# In[ ]:


print(res)


# In[ ]:


cleaned_train_df['content'][3]


# In[ ]:


cleaned_train_df['sentiment'][3]


# In[ ]:





# In[ ]:


cleaned_test_df['sentiment'].value_counts()


# In[ ]:


len(X_train[0])


# In[ ]:


invalid_roe = [index for index, row in enumerate(X_val) if len(row) > 50]

print(invalid_roe)


# In[ ]:


# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token=1)
# The oov_token is a placeholder token that replaces any OOV words during the text_to_sequence calls.
#This ensures that your model can handle new, unseen words gracefully.
tokenizer.fit_on_texts(X_train)


# In[ ]:


# On how many tweets did we train?
print(tokenizer.document_count)


# In[ ]:


# How many unique words?
len(tokenizer.word_counts)


# In[ ]:


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)


# In[ ]:


# Pad the sequences
max_length = 28  # Set max length for padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='post')


# In[ ]:


import numpy as np

# Create an embedding matrix
embedding_dim = 100  # Must match the dimension of FastText vectors
vocab_size = len(tokenizer.word_index) + 1   #it's the total number of unique words in your tokenizer’s vocabulary plus one. The +1 accounts for the padding token (index 0).
embedding_matrix = np.zeros((vocab_size, embedding_dim)) #embedding_matrix is initialized as a matrix of zeros with shape (vocab_size, embedding_dim). 
                                                        #This matrix will eventually hold the FastText vectors for each word in your vocabulary.

for word, i in tokenizer.word_index.items():
    if word in fasttext_model.wv:
        embedding_matrix[i] = fasttext_model.wv[word]

print(f"Embedding matrix shape: {embedding_matrix.shape}")


# In[ ]:


embedding_matrix


# ### RNN Model

# In[ ]:


#import keras as kp


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    weights=[embedding_matrix], 
                    input_length=max_length, 
                    trainable=False))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


import gensim
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# Assuming 'contents' is the column with the tweets and 'sentiment' is the label (0 or 1)
texts = cleaned_train_df['cleaned_contents'].astype(str).tolist() 
labels = cleaned_train_df['sentiment'].values

# Load pre-trained FastText embeddings
fasttext.util.download_model('en', if_exists='ignore')  # Download the model if not already present
ft = fasttext.load_model('cc.en.300.bin')  # Load the pre-trained FastText model

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# Pad sequences to ensure uniform input length
max_sequence_length = 50  # You can adjust this based on your data
X = pad_sequences(sequences, maxlen=max_sequence_length)

# Create an embedding matrix using the FastText embeddings
embedding_dim = 300  # FastText embeddings typically have 300 dimensions
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = ft.get_word_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=vocab_size, 
                    output_dim=embedding_dim, 
                    weights=[embedding_matrix], 
                    input_length=max_sequence_length, 
                    trainable=False))  # Set trainable to False to keep FastText embeddings static
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, 
          epochs=10, 
          batch_size=32, 
          validation_data=(X_val, y_val), 
          class_weight=class_weights_dict)

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')


