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


# The class designed for data preprocessing in NLP pipeline.
# Actions can be set to True or False following there need before feeding the data to the model.

class DataPreprocessor():
    def __init__(self, df, content_column, sentiment_column=None):
        """
        Initialize the DataPreprocessor class.
        
        Args commonly used in the Methods defined below:
        
        - df (pd.DataFrame): The DataFrame containing the dataset.
        
        - content_column (str): The name of the column containing the text content to analyze.
        
        - sentiment_column (str): The name of the column containing the sentiment labels.
            
        (other Args specific to defined methods are explained in that method's docstring)
        """
        self.df = df
        self.content_column = content_column
        self.sentiment_column = sentiment_column
    
    def remove_spam_content(self, similarity_threshold=0.80): # Removes extremely similar tweets since many bots post same tweet over and over.
        """
        Removes extremly similar samples using cosine similarity.
        
        Args:
        - similarity_threshold (float): (default=0.65) The threshold for cosine similarity above which samples are considered similar.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df[self.content_column])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        indices_to_remove = set()

        for i in range(len(cosine_sim)):
            for j in range(i + 1, len(cosine_sim)):
                if cosine_sim[i, j] > similarity_threshold:
                    indices_to_remove.add(i)
                    indices_to_remove.add(j)

        self.df = self.df.drop(index=indices_to_remove).reset_index(drop=True)


    def remove_hashtags(self, top_n=30): # whitelists top 30 hashtags and retains them in the tweet
        """
        Remove non-whitelisted hashtags from the tweet.
        
        Args:
        - top_n (int): Retain top n number of hashtags and remove others from the tweets.
        """
        
        def extract_hashtags(tweet):
            """Extract all hashtags from the text."""
            return re.findall(r'#\w+', tweet)
        
        def create_hashtag_whitelist(top_n):
            """Create a whitelist of the top N most frequent hashtags."""
            # Extract all hashtags from the dataset
            all_hashtags = sum(self.df[self.content_column].apply(extract_hashtags), [])
            # Count the occurrences of each hashtag
            hashtag_counts = Counter(all_hashtags)
            # Get the top N hashtags
            top_hashtags = [hashtag for hashtag, count in hashtag_counts.most_common(top_n)]
            return top_hashtags
        
        # Create the whitelist of top N hashtags
        whitelist = create_hashtag_whitelist(top_n=top_n)
        
        def clean_hashtags(tweet):
            hashtags = extract_hashtags(tweet)
            # Retain only hashtags in the whitelist
            for hashtag in hashtags:
                if hashtag not in whitelist:
                    tweet = tweet.replace(hashtag, '')
            return tweet
        
        # Apply the cleaning function to the dataframe
        column_to_use = 'cleaned_content' if 'cleaned_content' in self.df.columns else 'content'
        self.df['cleaned_content'] = self.df[column_to_use].apply(clean_hashtags)
    
    
    def remove_link(self): # Removes links from the tweet
        """
        Removes the links from the tweet
        """
        def remove_links(tweet):
            return re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        
        column_to_use = 'cleaned_content' if 'cleaned_content' in self.df.columns else 'content'
        self.df['cleaned_content'] = self.df[column_to_use].apply(remove_links)
    
    
    def remove_whitespace_html(self): # Removes whitespace and HTML from the tweets
        """
        Removes unnecessary whitespace and html markups from the text
        """
        def remove_whitesp(tweet):
            tweet = BeautifulSoup(tweet, 'html.parser').get_text()
            tweet = ' '.join(tweet.split())
            return tweet
        
        column_to_use = 'cleaned_content' if 'cleaned_content' in self.df.columns else 'content'
        self.df['cleaned_content'] = self.df[column_to_use].apply(remove_whitesp)
        
        
    def remove_emojis(self):
        """
        Retains and demojizes only expression related emojis
        """
        emoji_whitelist = [
            "😀", "😃", "😄", "😁", "😆", "😅", "😂", "😊", "😇", "🙂",
            "🙃", "😉", "😌", "😍", "😘", "😗", "😙", "😚", "😋", "😜",
            "😝", "😛", "🤑", "😎", "🤓", "😏", "😒", "😞", "😔", "😟",
            "😕", "🙁", "☹️", "😣", "😖", "😫", "😩", "😢", "😭", "😤",
            "😠", "😡", "🤬", "🤯", "😳", "🥵", "🥶", "😱", "😨", "😰",
            "😥", "😓", "🤗", "🤔", "🤭", "🤫", "🤥", "😶", "😐", "😑",
            "😬", "🙄", "😯", "😦", "😧", "😮", "😲", "😴", "🤤", "😪",
            "😵", "🤐", "🥴", "🤢", "🤮", "🤧", "😷", "🤒", "🤕", "🤑",
            "🤠", "😈", "👿", "👹", "👺", "🤡", "💩", "👻", "💀", "☠️",
            "😺", "😸", "😹", "😻", "😼", "😽", "🙀", "😿", "😾"
        ]
        
        # Pattern to match all emojis
        emoji_set = set(emoji.EMOJI_DATA.keys())
        
        def remove_and_demojize(tweet):
            tweet = ''.join(char for char in tweet if char in emoji_whitelist or char not in emoji_set)
            tweet = emoji.demojize(tweet, delimiters=(" ", " ")) # demojize the retained tweets to capture the sentiment
            return tweet
        
        column_to_use = 'cleaned_content' if 'cleaned_content' in self.df.columns else 'content'
        self.df['cleaned_content'] = self.df[column_to_use].apply(remove_and_demojize)
        

    def clean_text(self):
        """
        Cleans the tweet by removing punctuation, stopwords, and applying lemmatization.
        """
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        def clean(tweet):
            # Remove punctuation BUT RETAINS NUMBERS and $
            tweet = re.sub(r'[^a-zA-Z0-9\s$]', '', tweet)
            # Tokenize
            words = nltk.word_tokenize(tweet)
            # Remove stop words
            # words = [word for word in words if word.lower() not in stop_words]
            # Lemmatize words
            lemmatizer = nltk.WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word.lower()) for word in words]
            return ' '.join(words)
        
        #df['cleaned_content'] = df[content_column].apply(clean)
        column_to_use = 'cleaned_content' if 'cleaned_content' in self.df.columns else 'content'
        self.df['cleaned_content'] = self.df[column_to_use].apply(clean)
    
    
    def handle_class_imbalance_with_SMOTE(self):
        """
        Handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
        """
        
        tfidf = TfidfVectorizer(stop_words='english')
        X = tfidf.fit_transform(self.df[self.content_column])
        y = self.df[self.sentiment_column]
        
        # Apply SMOTE to the vectorized text
        smote = SMOTE(random_state=21)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.df = pd.DataFrame(X_resampled.toarray(), columns=tfidf.get_feature_names_out())
        self.df[self.sentiment_column] = y_resampled
    
    
    def preprocess(self, remove_spam=False, 
                   remove_hashtags=False, 
                   remove_link=False, 
                   remove_whitespace_html=False, 
                   remove_emoji=False, 
                   clean_text=False, 
                   balance_classes=False):
        """
        Performs the full preprocessing pipeline.

        Parameters:
        - remove_similar: bool (default=True)
            Whether to remove similar content.
        - balance_classes: bool (default=True)
            Whether to handle class imbalance.
        - clean_text: bool (default=True)
            Whether to clean text data.
        """
        if remove_spam:
            self.remove_spam_content()
        if remove_hashtags:
            self.remove_hashtags()
        if remove_link:
            self.remove_link()
        if remove_whitespace_html:
            self.remove_whitespace_html()
        if remove_emoji:
            self.remove_emojis()
        if balance_classes:
            self.handle_class_imbalance_with_SMOTE()
        if clean_text:
            self.clean_text()

        return self.df