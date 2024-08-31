#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! pip install pipreqs


# In[ ]:


#! pipreqs --encoding=utf-8 c:/Users/Diya/Documents/GItHub/DELTA_Element3


# # 0. Import Libraries 

# In[1]:


# For preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Data_Preprocess import DataPreprocessor

from Classification_Scores import ClassificationEvaluator

from Data_Augmentation import DataAugmentor

from DistilBERT import DistilBERTSentimentAnalyzer, run_pretrained_analysis, run_finetuned_analysis

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
from tensorflow.keras.metrics import AUC, Precision, Recall
from keras_tuner import BayesianOptimization

# For DistilBERT
# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline, TrainingArguments, Trainer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


# The **DataPreprocessor** class imported from **Data_Preprocess.py** is an all in one class with methods required for processing the raw tweets before feeding them to any model. Methods are defined individually to foster flexibility, allowing customization of data perprocessing steps per the requirement of specific model. At the same time, improves both, redability of the code and understandability of the underlying actions.
# 
# Below is the description of each method briefing its logic and reason.
# 
# 1. **remove_spam_content( )**<br>
# Tweets that are highly similar in their content (similarity>80%>) are removed. Bots often generate repeated content. I went through filtered out tweets and noticed that almost all of them were irrlevent to bitcoin and crypto.
# 
# 2. **remove_hashtags( )**<br> 
# Initially, I removed all hashtags from tweets, but later realized that complete removal is inappropriate as it causes relevant data loss. *This method retains top 30 most frequent hashtags as all of them are related to bitcoin and crypto. Noticed significant improvement in classification after its implementation.*
# 
# 3. **remove_link( )**<br>
# Removes link(s) from the tweet, if any.
# 
# 4. **remove_whitespace_html( )**<br>
# Removes all the whitespace and HTML tags from the tweet.
# 
# 5. **remove_emojis( )**<br>
# Similar t hashtahs, but *only expression and emotion related emojis are retained while all others removed. The retained emojis are then demojized as the add sentiment to the tweet.*
# 
# 6. **rlean( )**<br>
# This method removes punctuations and stop words and applies lemmatization. *It's worth to note that a significant performance improvement is observed when numbers and '$' character is retained in the tweet. This makes sense as the tweets are relaed to bitcoin and many of them have its price mentioned.*
# 

# The **ClassificationEvaluator** class imported from **Classification_Scores.py** is to evaluate the performance of a classification model particularly focusing on how well it handles imbalanced classes. It has methods defined are to print classification report, plot confusion matrix, ROC-AUC Curve and Precision-Recall curve.
# 
# 1. **evaluate( )**<br>
# Prints the classification report with precision, recall, and F1-score for each class, along with overall accuracy.
# 
# 2. **plot_confusion_matrix( )**<br>
# Plots a confusion matrix heatmap showing the count of true positive, true negative, false positive, and false negative predictions.
# 
# 3. **plot_roc_auc( )<br>**
# Plots the ROC curve and calculates the area under the curve (AUC). Helps to understand the trade-off between true positive rate (recall) and false positive rate (1-specificity) for different threshold settings.
# 
# 4. **plot_precision_recall_auc( )<br>**
# First of all, flips the labels to consider minority class for plotting. Calculates and plots the precision-recall curve for the minority class (negative class) and calculates the area under this curve (PR AUC).

# The **DataAugmentor** class imported from **Data_Augmentation.py** has the method 'augment_data' defined that takes the dataframe, and the number of samples as the argument, performs the augmentation of the minority class and returns the dataframe. It's implemented to tackle the problem of call imbalance.
# 
# Note: Since data augmentation is computationally expensive, I worked it out in colab and saved the augmented dataset in the directory with the name 'augmented train data.csv'. From there I imported it directly in the notebook to run the train experiments on RNN and fine tuned DistilBERT.
# 
# More on data augmentation and class imbalance at the end of the section 1.Data Preprocessing.

# # 1. Data Preprocessing

# ### 1.1. Importing and understanding the data.

# In[2]:


# Importing the train and test datasets
train_df = pd.read_parquet('btc_tweets_train.parquet.gzip').reset_index() # read_csv("augmented train data.csv")
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


# ### 1.2. A note on handling Class Imbalance

# Far as I have understood the dataset and this task, 'Call Imbalance' is the problem of utmost importance that's having significant influence on right from data preprocessing to performance evaluation and fine tuning. From all the available strategieg, below are a few that I found suiting the characteristics of our dataset and followed them to pacify the impact of class imbalance
# 
# 1. **Using different Metrics**<br>
# Accuracy is not the best metric in the presence of class imbalance. Hence throughout the notebook, considered other metrics like precision, recall, f1-score, ROC_AUC curve and PR curve.
# 
# 2. **Resampling Techniques**<br>
# - **Undersampling**: not preferred since the dataset is small and undersampling will result in loss of important information.
# - **Oversampling**: In this I particularly followed 2 approaches:-<br>
# **SMOTE** was rejected since it performs best with numerical data. I still vectorized the text and tested it, but the results were not impressive. Both recall and precision for minority (negative) class worsened.<br>
# **Data Augmentation** I used a pre-trained 'bert-base-uncased' model to generate additioanl 1000 minority class examples from the existing (270approx) samples. Trained RNN model with the 'augmented train df.csv' and below is its classification report and confusion matrix. Precision and Recall even after data augmentation is well below 50% although AUC shuowed slight improvement from 0.59 to 0.61. I think the reason is extremly limited number of minority samples to regenerate from. Hence the regenerated samples significantly lacked variation needed to train the model. <br>
# ![alt text](<augmented df classification perf.png>)
# ![alt text](<augmented df confusion matrix.png>)
# 
# 3. **Using Class Weights**
# Assigned higher weight to minority class so that the model pays more attention to them while training. Good starting point is the multiple of minority class such that 'minority class samples x multiple = majority class samples'. This is the most convenent way that yielded promising results.
# 
# **Colclusion:** With different in place, use of class weights was the most effective in improving the classification metrics for the negative class. Also it is the most convenient and flexible way to work with. I have followed this approach to help training with an imbalanced dataset.

# In the cell below is the code to perform data augmentation. However, since it was not giving any improved results, I havent used the augmented dataset for training. The code only sits to demonstrate the implementation. A copy of augmented data sits in the folder for review, if needed.

# In[21]:


# # Clean the tweets to get cleaned augmented tweets
# augmenter_data_preprocess = DataPreprocessor(df=train_df, content_column='content')
# augmented_train_df = augmenter_data_preprocess.preprocess(remove_spam=True, remove_link=True, remove_hashtags=True, remove_whitespace_html=True,remove_emoji=True)

# # To augment the dataset
# augmenter = DataAugmentor()
# augmented_train_df = augmenter.augment_data(df=augmented_train_df)
# augmented_train_df.to_csv('augmented train data.csv')


# # 2. Benchmark: vaderSentiment Dictionary

# For sentiment analysis using VADER, it's best to apply VADER to the raw, uncleaned text to leverage its strengths in handling informal language, punctuation, and emojis. However, when it comes to Links, it is best to remove them. Links are irrelevent to the sentiment and could add unnecessary noise, potentially influencing the sentiment. Also after multiple iterations, it was obseved that most of the emojis were adding no value but the noise. Hence only expression specific emojis are retained and ret all are removed. 

# The preprocessing pipeline for vaderSentiment
# - Clean the tweets
# - Then apply vader sentiment dictionary on the cleaned tweets
# - Initialize the vaderSentiment analyzer
# - Apply it on the cleaned tweets and get the compound scores
# - Set the threshold to transform the compound scores to positive class (1) or negative class (0)
# - Generate classification report

# ### 2.1. Clean the tweets for VaderSentiment

# In[8]:


vader_test_df = test_df.copy()

# Initialize the DataPreprocessor class for vaderSentiment
vader_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[9]:


# Getting the tweets ready by removing spam, unnecessary hashtags, links, whitespace, HTML tags and emojis
vader_test_df = vader_test_datapreprocessor.preprocess( remove_spam=True,
                                                        remove_hashtags=True, 
                                                        remove_link=True, 
                                                        remove_whitespace_html=True, 
                                                        remove_emoji=True )


# In[10]:


vader_test_df.head()


# In[11]:


# look over an example to see if the tweets are cleaned as expected
index = 43
vader_test_df['content'].iloc[index], vader_test_df['cleaned_content'].iloc[index], vader_test_df['sentiment'][index]


# ### 2.2. Fit the VaderSentiment on cleaned tweets in test data

# In[12]:


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


# In[13]:


vader_test_df.head()


# ### 2.3. Generate classification report for vaderSentiment

# In[14]:


# Initializing the classification performance evaluator class for vader
vader_performance_evaluator = ClassificationEvaluator(true_labels = vader_test_df['sentiment'], 
                                                      predicted_labels = vader_test_df['vader_sentiment_label'])

vader_performance_evaluator.evaluate()
vader_performance_evaluator.plot_confusion_matrix()


# 
# #### vaderSentiment Classification Report Interpretation
# 
# - VaderSentiment is better off in identifying positive sentiments.
# - Recall 92% indicates model captures most of the positive tweets.
# - This is expected anyways, since the dataset is imbalanced.

# # 3. RNN 

# ### 3.1. Getting the dataset ready for training the embeddings

# In[15]:


# Initialize the DataPreprocessor class for DistilBERT
rnn_train_datapreprocessor = DataPreprocessor(df=train_df, content_column='content')
rnn_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[16]:


# tweets in both, train and test datsasets are cleaned
rnn_train_df = rnn_train_datapreprocessor.preprocess(remove_spam=True, 
                                                     remove_hashtags=True, 
                                                     remove_link=True, 
                                                     remove_whitespace_html=True, 
                                                     remove_emoji=True, 
                                                     clean_text=True)

rnn_test_df = rnn_test_datapreprocessor.preprocess(remove_spam=True, 
                                                   remove_hashtags=True, 
                                                   remove_link=True, 
                                                   remove_whitespace_html=True, 
                                                   remove_emoji=True, 
                                                   clean_text=True)


# In[ ]:


# Exmple to check if cleaning is successful
i = 43
rnn_train_df['content'][i], rnn_train_df['cleaned_content'][i], rnn_train_df['sentiment'][i]


# In[ ]:


rnn_train_df.head()


# ### 3.2. Training the FastText embeddings

# In[ ]:


nltk.download('punkt')
nltk.download('wordnet')


# #### Parameter selection in the FastText training
# 
# Monitored model performance changing the following parameters and finally selected those that RNN permormed best with.
# 
# - vector_size=50:  Dimensionality of the word vectors (embeddings) that the model wil learn. RNN model performed best with the size of 50.
# 
# - window=5: Model will look at the 5 words before and after the target word in a sentence during training.
# 
# - min_count=1:Any word that appearing even once in the corpus will be included in the vocabulary. Set to 1 since the dataset is small.
# 
# - sg=0: 0 means that the model will use the Continuous Bag of Words (CBOW) algorithm, where the context (surrounding words) predicts the target word.<br>
# If sg=1, the model will use the Skip-gram algorithm, where the target word predivts the context.

# In[17]:


# Tokenize the cleaned tweets (split by spaces)
tokenized_tweets = [tweet.split() for tweet in rnn_train_df['cleaned_content']]

# Train FastText model using Gensim's implementation
fasttext_model = FastText(sentences=tokenized_tweets, vector_size=50, window=5, min_count=1, sg=0, epochs=10)


# In[18]:


# Example: Get vector for a word 
print(f"Vector for 'bitcoin': {fasttext_model.wv['bitcoin']}")

# Example: Get most similar words
print(f"Words similar to 'bitcoin': {fasttext_model.wv.most_similar('bitcoin')}")


# ### 3.3. Building and Training the RNN model

# In[19]:


# Tweet length plot to decide on sequence length.

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

# In[20]:


# Parameters
max_sequence_length = 50
embedding_dim = 50
embedding_matrix = fasttext_model.wv.vectors # embedding matrix to be passed as trained weights in the RNN embedding layer
vocab_size = len(fasttext_model.wv)


# In[21]:


# Tokenize the tweets
tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(rnn_train_df['cleaned_content'])

# sequences are tweets transformed into arrays where each word is sequentially replaced by a number which corresponds to the index of that word in the vocabulary
train_sequences = tokenizer.texts_to_sequences(rnn_train_df['cleaned_content']) 
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length, padding='post')

test_sequences = tokenizer.texts_to_sequences(rnn_test_df['cleaned_content'])
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length, padding='post')


# In[22]:


# An example look over
padded_test_sequences[89]


# In[23]:


# Dimension check
padded_train_sequences.shape, rnn_train_df['sentiment'].shape


# In[24]:


# Split the data to training and validation
X_train, X_val, y_train, y_val = train_test_split(padded_train_sequences, rnn_train_df['sentiment'], test_size = 0.2, random_state = 9)
# Changing the dtype from series to array
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()

# Preparing the test data
rnn_X_test = padded_test_sequences
rnn_y_test = rnn_test_df['sentiment'].to_numpy()


# - Class weights are calculated to instruct model to pay more attention to the minority class. Model performs fairly same when the weight for class 0 is incrementally changed from 2.6 to 3.1 after which it over predicts minority class imparing the model performance on majority class. 
# - It is finally dictated by the classification objective to decide on which class is more important. The weights will be ajusted accordingly. In this notebook, weights with which model performs the best on both the classes are retained.

# In[25]:


# Calculate the class weights to handle imbalance
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
class_weights
#class_weights = {0: 1, 1: 1}


# ### RNN Model Structure and Features:
# 
# 1. Input Layer <br>
# The model expects sequences of a specific length (max_sequence_length). This length corresponds to the number of tokens in each tweet i.e. text sequence.
# 
# 2. Embedding Layer<br>
# Vocabulary Size (vocab_size): The total number of unique tokens.<br>
# Embedding Dimension (embd_dim): Each word or token is represented as a vector of this length. Captures semantic information.<br>
# Pretrained Weights (embedding_matrix): The embeddings are initialized with pretrained weights, such as those from FastText. By setting trainable=False, you ensure these embeddings are not updated during training, preserving the pretrained semantic knowledge.
# RNN Layer:
# 
# 3. RNN Type (LSTM or GRU)<br> 
# The model allows for flexibility in choosing between Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU). 
# LSTM is preferred for handling long sequences, while GRU is faster and has fewer parameters.
# Units (64): The dimensionality of the output space for the RNN layer.
# Kernel Regularizer (l2(0.02)): Adds a penalty on the weights to prevent overfitting. Encourages the model to learn simpler patterns.
# Bidirectional Option:
# 
# 4. Bidirectional Layer (bidirectional=True/False)<br>
# If True, wraps the RNN layer in a Bidirectional wrapper, allowing the model to learn from the sequence in both forward and backward directions.
# Enhancinces the model's understanding of context.
# 
# 5. Dropout Layer<br>
# Dropout (0.2): Randomly sets 20% of the inputs to zero during training to prevent overfitting. Makes the model more robust.
# 
# 6. Output Layer<br>
# Dense Layer: A single unit with a sigmoid activation function suitable for binary classification tasks. Outputs probability for binary classification.

# In[26]:


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
    
    model.compile(optimizer=Adam(learning_rate=0.000196), loss='binary_crossentropy', metrics=['accuracy']) #, ['accuracy'], Precision(), , AUC(), Recall())
    return model


# In[27]:


# Initialize the model
model = build_rnn_model(embd_dim=embedding_dim, rnn_type='LSTM', bidirectional=False)
model.summary()


# In[28]:


# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=40, batch_size=12, validation_data=(X_val, y_val), class_weight=class_weights, callbacks=[early_stopping])


# In[29]:


# Evaluate the model
loss, accuracy = model.evaluate(rnn_X_test, rnn_y_test)


# In[30]:


# Predicting sentiments on new tweets
rnn_predictions = model.predict(rnn_X_test)
# Convert probabilities to binary predictions
rnn_predictions_discrete = (rnn_predictions >= 0.5).astype(int)  


# In[31]:


# # Initializing the performance evaluator class for RNN
rnn_performance_evaluator = ClassificationEvaluator(true_labels=rnn_y_test, 
                                                    predicted_labels=rnn_predictions_discrete, 
                                                    predicted_probs=rnn_predictions)

rnn_performance_evaluator.evaluate()
rnn_performance_evaluator.plot_confusion_matrix()
rnn_performance_evaluator.plot_roc_auc()
rnn_performance_evaluator.plot_precision_recall_auc()


# #### RNN Classification Report Interpretation
# 
# 1. Classification Report
# - Mdodel's performance on class 1 (positive) is fairly good. However, all the three metrics, precision, recall and f1-score on class 0 need attention. This can be atrributed to the imbalance in the dataset with class 1 having more instances (387) than class 0 (96). The performance would have been better, if the model would have had more negative samples to learn from.
# 
# 2. ROC_AUC Curve
# - The AUC of 0.58 suggests that the model has limited ability to distinguish between the positive and negative classes. Its marginally better than random guessing.
# 
# 3. PR Curve
# - Low AUC (0.17) suggests poor model performance in distinguishing between the positive and negative classes for minority class.<br>
# - Both precision and recall are low across different thresholds meaning a struggling model to identify minority class, resulting in may fasle positives and fasle negatives. Can be noticed in the confusion matrix.<br>

# ### 3.4. Hyperparameter tuning

# In[ ]:


def build_hypermodel(hp):
    model = Sequential()
    model.add(Input(shape=(max_sequence_length,)))
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim,
                        weights=[embedding_matrix],  
                        trainable=False))
    
    # Choose between LSTM and GRU
    rnn_type = hp.Choice('rnn_type', ['LSTM', 'GRU'])
    rnn_units = 64 #hp.Int('rnn_units', min_value=32, max_value=128, step=32)
    
    if rnn_type == 'LSTM':
        rnn_layer = LSTM(units=rnn_units, return_sequences=False, kernel_regularizer=l2(hp.Float('l2', 0.0, 0.1, step=0.01)))
    else:
        rnn_layer = GRU(units=rnn_units, return_sequences=False, kernel_regularizer=l2(hp.Float('l2', 0.0, 0.1, step=0.01)))
    
    # Add Bidirectional wrapper if specified
    if hp.Boolean('bidirectional'):
        model.add(Bidirectional(rnn_layer))
    else:
        model.add(rnn_layer)
    
    model.add(Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(hp.Float('l2_dense', 0.0, 0.1, step=0.01))))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-5, 1e-3, sampling='log')),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model


# In[ ]:


tuner = BayesianOptimization(
    build_hypermodel,
    objective='val_accuracy',
    max_trials=100,  # You can adjust this based on your computational resources
    executions_per_trial=1,  # Averages the results over multiple runs
    directory='bayesian_optimization',
    project_name='rnn_tuning'
)


# In[ ]:


X_val.shape


# In[ ]:


# Early stopping callback
hpt_early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# # Start the search
tuner.search(X_train, y_train,
             epochs=10,
             validation_data=(X_val, y_val),
             batch_size=32,
             callbacks=[hpt_early_stopping])


# # Get the best hyperparameters and model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

# Print the best hyperparameters
print(f"Best RNN type: {best_hps.get('rnn_type')}")
print(f"Best number of RNN units: {best_hps.get('rnn_units')}")
print(f"Use bidirectional: {best_hps.get('bidirectional')}")
print(f"Best dropout rate: {best_hps.get('dropout')}")
print(f"Best learning rate: {best_hps.get('learning_rate')}")

# Summary of the best model
best_model.summary()


# # 4. DistilBERT

# ### 4.1. Clean the tweets for DistilBERT

# In[8]:


# Initialize the DataPreprocessor class for DistilBERT
dtb_train_datapreprocessor = DataPreprocessor(df=train_df, content_column='content')
dtb_test_datapreprocessor = DataPreprocessor(df=test_df, content_column='content')


# In[9]:


dtb_train_df = dtb_train_datapreprocessor.preprocess(remove_spam=True, 
                                                     remove_hashtags=True, 
                                                     remove_emoji=True, 
                                                     remove_link=True, 
                                                     remove_whitespace_html=True)

dtb_test_df = dtb_test_datapreprocessor.preprocess(remove_spam=True, 
                                                   remove_hashtags=True, 
                                                   remove_emoji=True, 
                                                   remove_link=True, 
                                                   remove_whitespace_html=True)


# In[34]:


i = 7
dtb_test_df['content'][i], dtb_test_df['cleaned_content'][i], dtb_test_df['sentiment'][i]


# In[ ]:


dtb_test_df.head()


# In[10]:


# Split the data to training and validation
dtb_X_train, dtb_X_val, dtb_y_train, dtb_y_val = train_test_split(dtb_train_df['cleaned_content'], dtb_train_df['sentiment'], test_size = 0.2, random_state = 9)
# Changing the dtype from series to array
dtb_y_train = dtb_y_train #.to_numpy()
dtb_y_val = dtb_y_val #.to_numpy()

# Preparing the test data
dtb_X_test = dtb_test_df['cleaned_content']
dtb_y_test = dtb_test_df['sentiment']


# ### 4.2. Predict with pre-trained and fine-tuned DistilBERT

# The entire pipeline of generating predicitons using a pre-trained DistilBERT and then fine-tuned DistilBERT is implemented in a class 'DistilBERTSentimentAnalyzer' imported from 'DistilBERT.py'. Here is the detailed description of the approach.
# 
# #### **Pre-trained Analysis Pipeline:**
# 
# 1. **Tokenization:**
#    - 'pretrained_tokenizer' to tokenize and preprocess the input tweets.
#    - Convert the tweets into a format suitable for the model. Padds and truncates the tweets to specified maximum length.
# 
# 2. **Prediction:**
#    - Pass the tokenized input to the 'pretrained_model'.
#    - The model outputs logits, which represent the raw predictions.
#    - 'torch.argmax' to convert the logits into predicted class labels.
# 
# 3. **Evaluation:**
#    - Evaluate the performance of the pre-trained model on the test data using the predicted labels.
# 
# **Function:**
# - 'run_pretrained_analysis(analyzer, test_texts)'
# 
# #### **Fine-tuned Analysis Pipeline:**
# 
# 1. **Tokenization:**
#    - 'finetuned_tokenizer' to tokenize and preprocess the training, validation, and test tweets.
#    - Prepares the data for training and evaluation.
# 
# 2. **Prepare Labels:**
#    - Convert the training and validation labels into tensors.
# 
# 3. **Initial Training (with Frozen Layers):**
#    - Freeze all layers of the 'finetuned_model' except for the output layer.
#    - Set up training arguments using 'TrainingArguments'.
#    - Define training and validation datasets.
#    - Train the model using the 'Trainer' class, updating only the output layer initially.
# 
# 4. **Gradual Unfreezing:**
#    - Unfreeze all layers of the model to allow for further fine-tuning.
#    - Continue training the model with all layers unfrozen to adjust pre-trained features for the specific task.
# 
# 5. **Evaluation:**
#    - Evaluate the fine-tuned model on the test data using the updated weights.
# 
# **Function:**
# - 'run_finetuned_analysis(analyzer, train_texts, train_labels, val_texts, val_labels, test_texts)'

# In[11]:


# Initialize the analyzer
analyzer = DistilBERTSentimentAnalyzer(
    pretrained_model_name='DT12the/distilbert-sentiment-analysis',
    finetuned_model_name='distilbert-base-uncased-finetuned-sst-2-english'
)

# Example usage with your data
pt_dtb_predictions = run_pretrained_analysis(analyzer, dtb_test_df['cleaned_content'].to_list())

# # Fine-tune and evaluate
ft_dtb_predictions = run_finetuned_analysis(
    analyzer,
    train_texts = dtb_X_train.to_list(),
    train_labels = dtb_y_train.values,
    val_texts = dtb_X_val.to_list(),
    val_labels = dtb_y_val.values,
    test_texts= dtb_X_test.to_list()
)


# ### 4.3. Classification report for pre-trained DistilBERT

# In[12]:


# The pre-trained DistilBERT maps positive class to 0 and negative class to 1, which is opposite of our labels. Hence flipped them
flipped_predictions = 1 - pt_dtb_predictions

# Initializing the performance evaluator class for pre-trained DistilBERT
pt_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=flipped_predictions)

pt_dtb_performance_evaluator.evaluate()
pt_dtb_performance_evaluator.plot_confusion_matrix()


# ### 4.4. Classification report for fine-tuned DistilBERT

# In[13]:


# Initializing the performance evaluator class for pre-trained DistilBERT
ft_dtb_performance_evaluator = ClassificationEvaluator(true_labels=dtb_test_df['sentiment'].to_list(), 
                                                       predicted_labels=ft_dtb_predictions)

ft_dtb_performance_evaluator.evaluate()
ft_dtb_performance_evaluator.plot_confusion_matrix()


# # 5. Final Comparison

# ### 5.1. Code

# In[14]:


from sklearn.metrics import classification_report
import tabulate

def get_classification_scores(true_labels, predicted_labels):
    report = classification_report(true_labels, predicted_labels, output_dict=True)
    # Extract the relevant scores
    scores = {
        'precision_0': report['0']['precision'],
        'recall_0': report['0']['recall'],
        'f1_score_0': report['0']['f1-score'],
        'precision_1': report['1']['precision'],
        'recall_1': report['1']['recall'],
        'f1_score_1': report['1']['f1-score'],
    }
    return scores

model_predictions = {
    'vaderSentiment': vader_test_df['vader_sentiment_label'],
    'RNN (LSTM 64)': rnn_predictions_discrete, 
    'Pre-trained DistilBERT': flipped_predictions,
    'Fine-tuned DistilBERT': ft_dtb_predictions,  
}

# True labels
true_labels = dtb_test_df['sentiment'].to_list()

results = []

for model_name, predictions in model_predictions.items():
    scores = get_classification_scores(true_labels, predictions)
    scores['model'] = model_name
    results.append(scores)

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.set_index('model').map(lambda x: f"{x:.2f}")

# Convert DataFrame to table format
comparison_table = tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt='.3f')


# ### 5.2. Comparison Table

# In[15]:


print(comparison_table)


# - The table compares the classification performane of all 4 models tracking their precision, recall and F1 score for both the classes.
# - From the numbers, **Fine-tuned DistilBERT** is the best overall, consistently giving high scores in precision, recall, and F1 score for both categories. It outperforms other models, particularly the minority class, which is crucial in this imbalanced dataset.
