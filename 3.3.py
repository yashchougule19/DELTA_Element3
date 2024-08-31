#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! pip install pipreqs


# In[ ]:


#! pipreqs --encoding=utf-8 c:/Users/Diya/Documents/GItHub/DELTA_Element3


# # 0. Import Libraries 

# In[97]:


# For preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from DataPreprocess import DataPreprocessor

from ClassificationScores import ClassificationEvaluator

# For VaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# For RNN model 
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

# For DistilBERT
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# # 1. Data Preprocessing

# ### 1.1. Importing and understanding the data.

# In[2]:


# Importing the train and test datasets
train_df = pd.read_parquet('btc_tweets_train.parquet.gzip').reset_index()
test_df = pd.read_parquet('btc_tweets_test.parquet.gzip').reset_index()


# In[3]:


# Running a few checks to understand the datasets (shape, empty tweets, info and head)
def df_info(df):
    print(f'shape of the datafreme: {df.shape}')
    print(f'Number of empty values: {df.isnull().sum().sum()}')
    print(df.info())
    print(df.head())


# In[4]:


df_info(train_df)


# In[5]:


df_info(test_df)


# In[6]:


# Dropping the unnecessary features.
# Note: Although 'hashtags' feature is removed, hashtags in the tweet are differently handled in DataPreprocess class
train_df = train_df.drop(['tweet ID', 'user_displayname', 'hashtags'], axis=1)
test_df = test_df.drop(['tweet ID', 'user_displayname', 'hashtags'], axis=1)


# In[7]:


# Converting the sentiment labels from bool to int
train_df['sentiment'] = train_df['sentiment'].astype(int)
test_df['sentiment'] = test_df['sentiment'].astype(int)


# The DataPreprocessor class imported from DataPreprocess.py is an all in one class that has various methods required for processing the raw tweets before feeding them to any model. Methods are defined individually to foster flexibility, allowing customization of data perprocessing steps per the advantages and limitations of the specific model. At the same time, doing so improves both, redability of the code and understandability of the underlying process.
# 
# Below is the description of each method briefing the logic and reason.
# 
# 1. **Remove_spam_content( )**<br>
# Tweets that are highly similar in their content (similarity>80%>) are removed. Bots often generate repeated content. I went through filtered out tweets and noticed that almost all of them were irrlevent to bitcoin and crypto.
# 
# 2. **Remove_hashtags( )**<br> 
# Initially, I removed all hashtags from tweets, but later realized that complete removal is inappropriate as it causes relevant data loss. *This method retains top 30 most frequent hashtags as all of them are related to bitcoin and crypto. Noticed significant improvement in classification after its implementation.*
# 
# 3. **Remove_link( )**<br>
# Removes link(s) from the tweet, if any.
# 
# 4. **Remove_whitespace_html( )**<br>
# Removes all the whitespace and HTML tags from the tweet.
# 
# 5. **Remove_emojis( )**<br>
# Similar t hashtahs, but *only expression and emotion related emojis are retained while all others removed. The retained emojis are then demojized as the add sentiment to the tweet.*
# 
# 6. **Clean( )**<br>
# This method removes punctuations and stop words and applies lemmatization. *It's wrth to note that a significant performance improvement is observed when numbers and '$' character is retained in the tweet. This makes sense as the tweets are relaed to bitcoin and many of them have its price mentioned.*
# 

# ### A few trials

# In[8]:


# def extract_special_characters(text):
#     # Regular expression to match special characters
#     return re.findall(r'[^a-zA-Z0-9\s]', text)

# # Extract special characters from the selected column (e.g., 'content')
# df = train_df.copy()
# df['special_chars'] = df['content'].apply(extract_special_characters)

# # Flatten the list of special characters and count their frequency
# special_chars_list = df['special_chars'].sum()
# special_chars_count = Counter(special_chars_list)

# # Get the most common special characters
# most_common_special_chars = special_chars_count.most_common()

# # Display the results
# for char, count in most_common_special_chars:
#     print(f"'{char}': {count}")


# In[9]:


# def extract_emojis(text):
#     return ''.join(char for char in text if char in emoji.EMOJI_DATA)

# # Extract emojis from the selected column (e.g., 'content')
# #column_to_use = 'cleaned_content' if 'cleaned_content' in df.columns else 'content'
# df = train_df.copy()
# df['emojis'] = df['content'].apply(extract_emojis)

# # Flatten the list of emojis and count the frequency of each emoji
# emoji_list = df['emojis'].sum()
# emoji_count = Counter(emoji_list)

# # Get the top 50 most used emojis
# top_50_emojis = emoji_count.most_common(200)

# # Display the results
# for emoji_char, count in top_50_emojis:
#     print(f"{emoji_char}: {count}")


# **NOTE: both the cleaned datasets above are still imbalanced with True values largly outnumbered than False. The imbalance needs to be taken care of by assigning class weights dring model training.**

# # 2. Benchmark: vaderSentiment Sentiment Dictionary

# For sentiment analysis using VADER, it's best to apply VADER to the raw, uncleaned text to leverage its strengths in handling informal language, punctuation, and emojis. However, when it comes to Links, it is best to remove them. Links are irrelevent to the sentiment and could add unnecessary noise, potentially influencing the sentiment. Also after multiple iterations, it was obseved that most of the emojis are adding no value but noise. Hence only expression specific emojis are retained and ret all are removed. 

# The preprocessing flow for vaderSentiment
# - Remove spam tweets
# - Remove links
# - Remove unnecessary hashtags
# - Remove whitespace and HTML tags
# - Remove unnecessary emojis
# - Then apply vader sentiment dictionary on the cleaned tweets

# ### 2.1. Getting the dataset ready for VaderSentiment

# In[ ]:


vader_test_df = test_df.copy()

# Initialize the DataPreprocessor class for vaderSentiment
vader_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[ ]:


# Getting the dataset ready by removing spam, unnecessary hashtags, links, whitespace, HTML tags and emojis
vader_test_df = vader_test_datapreprocessor.preprocess( remove_spam=True,
                                                        remove_hashtags=True, 
                                                        remove_link=True, 
                                                        remove_whitespace_html=True, 
                                                        remove_emoji=True )


# In[ ]:


vader_test_df.head()


# In[ ]:


# look over an example to see if the tweets are cleaned as expected
index = 43
vader_test_df['content'].iloc[index], vader_test_df['cleaned_content'].iloc[index], vader_test_df['sentiment'][index]


# ### 2.2. Fitting the VaderSentiment on cleaned test data

# In[ ]:


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to apply VADER sentiment analysis and get the compound score
def get_sentiment_score(text):
    sentiment_dict = analyzer.polarity_scores(text)
    return sentiment_dict['compound']  # 'compound' score is a normalized score between -1 (negative) and +1 (positive)

# Apply sentiment analysis to the 'content' column and create a new column for the sentiment score
vader_test_df['vader_sentiment'] = vader_test_df['cleaned_content'].apply(get_sentiment_score)

# Classify the sentiment based on the compound score.
# Threshold of -0.05 results in the best classification performance
def classify_sentiment(score):
    if score >= -0.05:
        return True
    else:
        return False

# Apply the classification and create a new column for the sentiment label
vader_test_df['vader_sentiment_label'] = vader_test_df['vader_sentiment'].apply(classify_sentiment).astype(int)


# In[ ]:


vader_test_df.head()


# ### 2.3. Evaluating the vaderSentiment classification performance

# In[ ]:


# Initializing the classification performance evaluator class for vader
vader_performance_evaluator = ClassificationEvaluator(true_labels = vader_test_df['sentiment'], 
                                                      predicted_labels = vader_test_df['vader_sentiment_label'])

vader_performance_evaluator.evaluate()
vader_performance_evaluator.plot_confusion_matrix()


# - VaderSentiment is better off in identifying positive sentiments.
# - Recall 92% indicates model captures most positive instances.
# - This is expected anyways, since the dataset is imbalanced.

# # 3. RNN 

# ### 3.1. Getting the dataset ready for training the embeddings

# In[98]:


# Initialize the DataPreprocessor class for DistilBERT
rnn_train_datapreprocessor = DataPreprocessor(df=train_df, content_column='content')
rnn_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[99]:


rnn_train_df = rnn_train_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_link=True, remove_whitespace_html=True, remove_emoji=True, clean_text=True, balance_classes=False)
rnn_test_df = rnn_test_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_link=True, remove_whitespace_html=True, remove_emoji=True, clean_text=True, balance_classes=False)


# In[ ]:


i = 83
rnn_train_df['content'][i], rnn_train_df['cleaned_content'][i], rnn_train_df['sentiment'][i]


# In[ ]:


rnn_train_df.head()


# ### 3.2. Training the FastText embeddings

# In[ ]:


nltk.download('punkt')
nltk.download('wordnet')


# In[100]:


# Tokenize the cleaned tweets (split by spaces)
tokenized_tweets = [tweet.split() for tweet in rnn_train_df['cleaned_content']]

# Train FastText model using Gensim's implementation
fasttext_model = FastText(sentences=tokenized_tweets, vector_size=50, window=5, min_count=1, sg=0, epochs=10)

# Save the model
#fasttext_model.save("fasttext_vs50.model")

# Load the model (for future use)
#fasttext_model = FastText.load("fasttext_vs50.model")


# In[ ]:


# Example: Get vector for a word 
print(f"Vector for 'bitcoin': {fasttext_model.wv['bitcoin']}")

# Example: Get most similar words
print(f"Words similar to 'bitcoin': {fasttext_model.wv.most_similar('bitcoin')}")


# ### 3.3. Building and Training the RNN model

# Tweet length plot to decide on sequence length.

# In[101]:


# List of tweets to be used for rnn training
tweets = rnn_train_df['cleaned_content'].to_list()

# Tokenize each tweet to calculate its length
tweet_lengths = [len(word_tokenize(tweet)) for tweet in tweets]

# Plot for the tweet length distribution
plt.figure(figsize=(6, 3))
sns.histplot(tweet_lengths, bins=20, kde=True, color='grey')
plt.title('Tweet Length Distribution')
plt.xlabel('Number of Tokens')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Sequence length of 50 farly covers all the tweets.

# In[102]:


# Parameters
max_sequence_length = 50
embedding_dim = 50
embedding_matrix = fasttext_model.wv.vectors # Create embedding matrix
vocab_size = len(fasttext_model.wv)


# In[ ]:


#fasttext_model.wv.vector_size, len(tokenizer.word_counts), vocab_size, tokenizer.word_index.items()


# In[103]:


# Tokenize the tweets
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(rnn_train_df['cleaned_content'])

# sequences are tweets transformed into arrays where each word is sequentially replaced by a number which corresponds to the index of that word in the vocabulary
train_sequences = tokenizer.texts_to_sequences(rnn_train_df['cleaned_content']) 
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(rnn_test_df['cleaned_content'])
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')


# In[ ]:


padded_test_sequences


# In[ ]:


# Dimension check
padded_train_sequences.shape, rnn_train_df['sentiment'].shape


# In[104]:


# Split the data to training and validation
X_train, X_val, y_train, y_val = train_test_split(padded_train_sequences, rnn_train_df['sentiment'], test_size = 0.2, random_state = 9)
# Changing the dtype from series to array
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Preparing the test data
rnn_X_test = padded_test_sequences
rnn_y_test = rnn_test_df['sentiment'].to_numpy()


# In[105]:


# Calculate the class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
class_weights


# Study about kernel_regularizer, learning_rate

# In[106]:


#from tensorflow.keras.metrics import AUC, Precision, Recall

# Build the RNN model
def build_rnn_model(embd_dim, rnn_type='LSTM', bidirectional=False):
    model = Sequential()
    model.add(Input(shape=(max_sequence_length,)))
    model.add(Embedding(input_dim = vocab_size, 
                        output_dim = embd_dim, 
                        weights=[embedding_matrix],  
                        trainable=False))
    
    if rnn_type == 'LSTM':
        rnn_layer = LSTM(units = 64, return_sequences = False, kernel_regularizer=l2(0.02))
    elif rnn_type == 'GRU':
        rnn_layer = GRU(units = 64, return_sequences= False, kernel_regularizer=l2(0.02))
        
    if bidirectional:
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)
        
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid', kernel_regularizer=l2(0.02)))
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy']) #, Precision(), Recall(), AUC()])
    return model


# In[118]:


# Initialize the model
model = build_rnn_model(embd_dim=embedding_dim, rnn_type='LSTM', bidirectional=False)
model.summary()


# In[123]:


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])


# In[124]:


# Evaluate the model
loss, accuracy = model.evaluate(rnn_X_test, rnn_y_test)


# In[125]:


# Predicting sentiments on new tweets
rnn_predictions = model.predict(rnn_X_test)
# Convert probabilities to binary predictions
rnn_predictions_discrete = (rnn_predictions >= 0.5).astype(int)  


# In[ ]:


# u = 24
# rnn_predictions[u], rnn_predictions_discrete[u], rnn_y_test[u], rnn_test_df['cleaned_content'].iloc[u]


# In[ ]:


#rnn_predictions


# In[126]:


# Initializing the performance evaluator class for RNN
rnn_performance_evaluator = ClassificationEvaluator(true_labels=rnn_y_test, 
                                                    predicted_labels=rnn_predictions_discrete, 
                                                    predicted_probs=rnn_predictions)

rnn_performance_evaluator.evaluate()
rnn_performance_evaluator.plot_confusion_matrix()
rnn_performance_evaluator.plot_roc_auc()


# # 4. DistilBERT

# ### 4.1. Data preprocessing: DistilBERT

# In[ ]:


# Initialize the DataPreprocessor class for DistilBERT
dtb_train_datapreprocessor = DataPreprocessor(df=train_df, content_column='content')
dtb_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[ ]:


dtb_train_df = dtb_train_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_emoji=True, remove_link=True, remove_whitespace_html=True)
dtb_test_df = dtb_test_datapreprocessor.preprocess(remove_spam=True, remove_hashtags=True, remove_emoji=True, remove_link=True, remove_whitespace_html=True)


# In[ ]:


i = 7
dtb_test_df['content'][i], dtb_test_df['cleaned_content'][i], dtb_test_df['sentiment'][i]


# In[ ]:


dtb_test_df.head()


# In[ ]:


# Split the data to training and validation
dtb_X_train, dtb_X_val, dtb_y_train, dtb_y_val = train_test_split(dtb_train_df['cleaned_content'], dtb_train_df['sentiment'], test_size = 0.2, random_state = 9)
# Changing the dtype from series to array
dtb_y_train = dtb_y_train #.to_numpy()
dtb_y_val = dtb_y_val #.to_numpy()

# Preparing the test data
dtb_X_test = dtb_test_df['cleaned_content']
dtb_y_test = dtb_test_df['sentiment']


# In[ ]:


from transformers import (AutoTokenizer, AutoModelForSequenceClassification, 
                          DistilBertTokenizer, DistilBertForSequenceClassification, 
                          pipeline, Trainer, TrainingArguments)
from torch.utils.data import Dataset

# Define a Dataset class
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# A class for handling pre-trained and fine-tuned DistilBERT models
class DistilBERTSentimentAnalyzer:
    def __init__(self, pretrained_model_name, finetuned_model_name, max_length=32):
        self.pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.pretrained_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
        self.finetuned_tokenizer = DistilBertTokenizer.from_pretrained(finetuned_model_name)
        self.finetuned_model = DistilBertForSequenceClassification.from_pretrained(finetuned_model_name)
        self.max_length = max_length

    def tokenize_data(self, texts, tokenizer):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

    def predict(self, tokens, model):
        with torch.no_grad():
            outputs = model(**tokens)
        return torch.argmax(outputs.logits, dim=1)

    def evaluate_model(self, model, tokens):
        with torch.no_grad():
            outputs = model(**tokens)
        return torch.argmax(outputs.logits, dim=1)

    def fine_tune(self, train_tokens, train_labels, val_tokens, val_labels, epochs=1, batch_size=16):
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir='./logs',
        )

        # Define training and evaluation datasets
        train_dataset = SentimentDataset(train_tokens, train_labels)
        val_dataset = SentimentDataset(val_tokens, val_labels)

        # Train DistilBERT model
        trainer = Trainer(
            model=self.finetuned_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )

        trainer.train()

# Utility functions for the analysis
def run_pretrained_analysis(analyzer, test_texts):
    tokens = analyzer.tokenize_data(test_texts, analyzer.pretrained_tokenizer)
    return analyzer.predict(tokens, analyzer.pretrained_model)

def run_finetuned_analysis(analyzer, train_texts, train_labels, val_texts, val_labels, test_texts):
    #X_train, y_train, X_val, y_val = train_test_split(train_texts, test_size=0.2)
    train_tokens = analyzer.tokenize_data(train_texts, analyzer.finetuned_tokenizer)
    val_tokens = analyzer.tokenize_data(val_texts, analyzer.finetuned_tokenizer)
    test_tokens = analyzer.tokenize_data(test_texts, analyzer.finetuned_tokenizer)

    # Convert labels to tensors
    train_labels_tensor = torch.tensor(list(map(int, train_labels)), dtype=torch.long)
    val_labels_tensor = torch.tensor(list(map(int, val_labels)), dtype=torch.long)

    # Fine-tune the model
    analyzer.fine_tune(train_tokens, train_labels_tensor, val_tokens, val_labels_tensor)

    # Evaluate on the test set
    return analyzer.evaluate_model(analyzer.finetuned_model, test_tokens)
    


# In[ ]:


dtb_y_train.values


# In[ ]:


# Initialize the analyzer
analyzer = DistilBERTSentimentAnalyzer(
    pretrained_model_name='DT12the/distilbert-sentiment-analysis',
    finetuned_model_name='distilbert-base-uncased-finetuned-sst-2-english'
)

# Example usage with your data
pt_dtb_predictions = run_pretrained_analysis(analyzer, dtb_test_df['cleaned_content'].to_list())

# Fine-tune and evaluate
ft_dtb_predictions = run_finetuned_analysis(
    analyzer,
    train_texts = dtb_X_train.to_list(),
    train_labels = dtb_y_train.values,
    val_texts = dtb_X_val.to_list(),
    val_labels = dtb_y_val.values,
    test_texts= dtb_X_test.to_list()
)


# In[ ]:


# Initializing the performance evaluator class for pre-trained DistilBERT
pt_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=pt_dtb_predictions)

pt_dtb_performance_evaluator.evaluate()
pt_dtb_performance_evaluator.plot_confusion_matrix()


# In[ ]:


# Initializing the performance evaluator class for pre-trained DistilBERT
ft_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=ft_dtb_predictions)

ft_dtb_performance_evaluator.evaluate()
ft_dtb_performance_evaluator.plot_confusion_matrix()


# ### 4.2. Pre-trained DistilBERT

# In[ ]:


pt_dtb_tokenizer = AutoTokenizer.from_pretrained('DT12the/distilbert-sentiment-analysis')
pt_dtb_model = AutoModelForSequenceClassification.from_pretrained('DT12the/distilbert-sentiment-analysis')


# In[ ]:


pt_dtb_test_tokens = pt_dtb_tokenizer(text=dtb_test_df['cleaned_content'].to_list(),
                                 padding=True,
                                 truncation=True,
                                 max_length=32,
                                 return_tensors='pt')


# In[ ]:


type(pt_dtb_test_tokens)


# In[ ]:


with torch.no_grad():
    pt_dtb_predictions = pt_dtb_model(**pt_dtb_test_tokens)


# In[ ]:


pt_dtb_predictions_discrete = torch.argmax(pt_dtb_predictions.logits, dim=1)


# In[ ]:


# Initializing the performance evaluator class for pre-trained DistilBERT
pt_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=pt_dtb_predictions_discrete)

pt_dtb_performance_evaluator.evaluate()
pt_dtb_performance_evaluator.plot_confusion_matrix()


# ### 4.3. Fine-tuned DistilBERT

# In[ ]:


ft_dtb_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
ft_dtb_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')


# In[ ]:


sentiment_analysis = pipeline('sentiment-analysis',
                              model = ft_dtb_model,
                              tokenizer = ft_dtb_tokenizer,
                              batch_size = 16)


# In[ ]:


# Tokenize the training and validation set
ft_dtb_train_tokens = ft_dtb_tokenizer(
    dtb_train_df['cleaned_content'].to_list(),
    padding=True,
    truncation=True,
    max_length=32,  # Adjust this according to your sequence lengths
    return_tensors='pt'
)

ft_dtb_test_tokens = ft_dtb_tokenizer(
    dtb_test_df['cleaned_content'].to_list(),
    padding=True,
    truncation=True,
    max_length=32,  # ⚠️ Adjust this if your sequences are longer or shorter, and explain why we did it that way
    return_tensors='pt'
)


# In[ ]:


# Convert the labels to a tensor
labels_train = torch.tensor(list(map(int, dtb_train_df['sentiment'].values)), dtype=torch.long)
labels_val = torch.tensor(list(map(int, dtb_test_df['sentiment'].values)), dtype=torch.long)


# In[ ]:


# Define a dataset class (optional, for clarity)
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[ ]:


##### Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=16,   # batch size for evaluation
    logging_dir='./logs',            # directory for storing logs
)


# In[ ]:


# Define training and evaluation datasets
train_dataset = SentimentDataset(ft_dtb_train_tokens, labels_train)
val_dataset = SentimentDataset(ft_dtb_test_tokens, labels_val) 


# In[ ]:


# Train DistilBERT model
trainer = Trainer(
    model=ft_dtb_model,              # the instantiated 🤗 Transformers model to be trained
    args=training_args,              # training arguments, defined above
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()


# In[ ]:


# Evaluate model on the test set
# Disable gradient calculation for evaluation
with torch.no_grad():
    ft_dtb_predictions = ft_dtb_model(**ft_dtb_test_tokens)

# Get the predicted class (0 or 1)
ft_dtb_predictions_discrete = torch.argmax(ft_dtb_predictions.logits, dim=1)

# Convert predictions to NumPy array for comparison
# predictions_DTB = predictions_DTB.numpy()


# In[ ]:


# Initializing the performance evaluator class for pre-trained DistilBERT
ft_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=ft_dtb_predictions_discrete)

ft_dtb_performance_evaluator.evaluate()
ft_dtb_performance_evaluator.plot_confusion_matrix()

