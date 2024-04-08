import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


class Recommender:

    def __init__(self, json_data=None, user=None):
        self.data = pd.DataFrame(json_data)
        self.user = user
        self.user_keywords = user['user_keywords']
        self.user_categories = user['user_categories']

        print("Loading GloVe model...")
        self.vectors = self.load_glove_model(path='gloves/converted_vectors300.txt')
        print("GloVe model loaded.")

    def delete_missing_values(self):
        # Check for NaN, missing, or NaT values
        missing_rows = self.data[self.data.isnull().any(axis=1)]

        if not missing_rows.empty:
            print("Deleting rows with missing values...")
            self.data.dropna(inplace=True)
            print("Rows with missing values have been deleted.")
        else:
            print("There are no missing values in the .DataFrame.")

    def text_preprocessing(self):

        def _remove_non_ascii(text):
            return ''.join(i for i in text if ord(i) < 128)
        
        def make_lower_case(text):
            return text.lower()
        
        def remove_stop_words(text):
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)
            return text
        
        def remove_html(text):
            html_pattern = re.compile('<.*?>')
            return html_pattern.sub(r'', text)
        
        def remove_punctuation(text):
            tokenizer = RegexpTokenizer(r'\w+')
            text = tokenizer.tokenize(text)
            text = " ".join(text)
            return text
        
        def remove_digits(text):
            return re.sub(r'\d+', '', text)
        
        self.data['title'] = self.data['title'].apply(_remove_non_ascii)
        self.data['title'] = self.data['title'].apply(make_lower_case)
        self.data['title'] = self.data['title'].apply(remove_stop_words)
        self.data['title'] = self.data['title'].apply(remove_html)
        self.data['title'] = self.data['title'].apply(remove_punctuation)
        self.data['title'] = self.data['title'].apply(remove_digits)

        self.data['short_description'] = self.data['short_description'].apply(_remove_non_ascii)
        self.data['short_description'] = self.data['short_description'].apply(make_lower_case)
        self.data['short_description'] = self.data['short_description'].apply(remove_stop_words)
        self.data['short_description'] = self.data['short_description'].apply(remove_html)
        self.data['short_description'] = self.data['short_description'].apply(remove_punctuation)
        self.data['short_description'] = self.data['short_description'].apply(remove_digits)

        self.data['description'] = self.data['description'].apply(make_lower_case)
        self.data['description'] = self.data['description'].apply(remove_stop_words)
        self.data['description'] = self.data['description'].apply(_remove_non_ascii)
        self.data['description'] = self.data['description'].apply(remove_html)
        self.data['description'] = self.data['description'].apply(remove_punctuation)
        self.data['description'] = self.data['description'].apply(remove_digits)
    
    def load_glove_model(self, path='gloves/converted_vectors300.txt'):
        self.vectors = KeyedVectors.load_word2vec_format(path, binary=False)

    def get_average_word2vec(self, tokens_list, generate_missing=False, k=300):
        if len(tokens_list) < 1:
            return np.zeros(k)
        
        vectorized = []
        for word in tokens_list:
            if word in self.vectors:
                vectorized.append(self.vectors[word])
            elif generate_missing:
                vectorized.append(np.random.rand(k))
        
        if not vectorized:
            return np.zeros(k)
        
        # Only compute the mean if there are non-zero vectors
        if len(vectorized) > 0:
            return np.mean(vectorized, axis=0)
        else:
            return np.zeros(k)

    def recommend_news_based_on_keyword(self, keyword):
        keyword = keyword.split()
        keyword = self.get_average_word2vec(keyword, self.vectors)
        
        # Calculate vectors for title
        # cut down the title into lists of words before applying the get_average_word2vec functio
        self.data['title_vector'] = self.data['title'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['short_desc_vector'] = self.data['short_description'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['desc_vector'] = self.data['description'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))

            # Compute cosine similarity for each title vector
        self.data['title_similarity'] = self.data.apply(lambda row: cosine_similarity([row['title_vector']], [keyword])[0][0], axis=1)
        self.data['short_desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['short_desc_vector']], [keyword])[0][0], axis=1)
        self.data['desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['desc_vector']], [keyword])[0][0], axis=1)

        # Combine similarities with weights
        title_weight = 0.7
        short_desc_weight = 0.2
        desc_weight = 0.1

        self.data['combined_similarity'] = (title_weight * self.data['title_similarity'] +
                                                short_desc_weight * self.data['short_desc_similarity'] + 
                                                desc_weight * self.data['desc_similarity'])

            # Sort by title similarity
        self.data = self.data.sort_values(by='title_similarity', ascending=False)
            
        return self.data

    def recommend_news_based_on_author(self, author, vectors, recent=False):
        author = author.split()
        author = self.get_average_word2vec(author, vectors)
        
        # Calculate vectors for title
        # cut down the title into lists of words before applying the get_average_word2vec functio
        self.data['author_vector'] = self.data['author'].apply(lambda x: self.get_average_word2vec(x.split(), vectors))

        # Compute cosine similarity for each title vector
        self.data['author_similarity'] = self.data.apply(lambda row: cosine_similarity([row['author_vector']], [author])[0][0], axis=1)

        # Sort by title similarity
        self.data = self.data.sort_values(by='author_similarity', ascending=False)

        # Sort by date_created if recent is True
        if recent:
            self.data = self.data.sort_values(by=['author_similarity', 'date_created'], ascending=[False, False])

        return self.data

    def recommend_news_based_on_keyword_and_preferences(self, user_data):
        user_keywords = user_data.get('user_keywords', [])
        user_categories = user_data.get('user_categories', [])

        if not user_keywords:
            return pd.DataFrame()  # Return an empty DataFrame if no user keywords are provided

        keyword = ' '.join(user_keywords)
        keyword_vector = self.get_average_word2vec(keyword.split(), self.vectors)

        # Calculate vectors for title, short description, and description
        self.data['title_vector'] = self.data['title'].apply(lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['short_desc_vector'] = self.data['short_description'].apply(
            lambda x: self.get_average_word2vec(x.split(), self.vectors))
        self.data['desc_vector'] = self.data['description'].apply(
            lambda x: self.get_average_word2vec(x.split(), self.vectors))

        # Compute cosine similarity for each feature
        self.data['title_similarity'] = self.data.apply(
            lambda row: cosine_similarity([row['title_vector']], [keyword_vector])[0][0], axis=1)
        self.data['short_desc_similarity'] = self.data.apply(
            lambda row: cosine_similarity([row['short_desc_vector']], [keyword_vector])[0][0], axis=1)
        self.data['desc_similarity'] = self.data.apply(
            lambda row: cosine_similarity([row['desc_vector']], [keyword_vector])[0][0], axis=1)

        # Combine similarities with weights
        title_weight = 0.7
        short_desc_weight = 0.2
        desc_weight = 0.1

        self.data['combined_similarity'] = (title_weight * self.data['title_similarity'] +
                                            short_desc_weight * self.data['short_desc_similarity'] +
                                            desc_weight * self.data['desc_similarity'])

        # Sort by combined similarity
        self.data = self.data.sort_values(by='combined_similarity', ascending=False)

        # Prioritize articles based on user preferences for categories
        if user_categories:
            category_weights = user_categories
            category_max_score = max(category_weights.values(), key=lambda x: x['score'])['score']
            for category, weight_info in category_weights.items():
                weight = weight_info['score']
                self.data.loc[self.data['category'] == category, 'combined_similarity'] *= (weight / category_max_score)

        # Sort again by adjusted combined similarity
        self.data = self.data.sort_values(by='combined_similarity', ascending=False)

        return self.data

    def recommend_news_based_on_keywords_and_preferences(self, vectors, user_preferences=None):
        # Initialize combined similarity column
        self.data['combined_similarity'] = 0.0
        
        # Iterate through each keyword and its score
        for keyword, score_info in self.user_keywords.items():
            score = score_info['score']
            # Calculate average word2vec representation for the keyword
            keyword_vec = self.get_average_word2vec(keyword.split(), vectors)
            
            # Calculate cosine similarity for each feature with the keyword
            self.data['title_similarity'] = self.data.apply(lambda row: cosine_similarity([row['title_vector']], [keyword_vec])[0][0], axis=1)
            self.data['short_desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['short_desc_vector']], [keyword_vec])[0][0], axis=1)
            self.data['desc_similarity'] = self.data.apply(lambda row: cosine_similarity([row['desc_vector']], [keyword_vec])[0][0], axis=1)

            # Combine similarities with weights
            title_weight = 0.7
            short_desc_weight = 0.2
            desc_weight = 0.1

            self.data['combined_similarity'] += (title_weight * self.data['title_similarity'] +
                                            short_desc_weight * self.data['short_desc_similarity'] +
                                            desc_weight * self.data['desc_similarity']) * score
        
        # Sort by combined similarity
        self.data = self.data.sort_values(by='combined_similarity', ascending=False)

        self.data = self.data.sort_values(by='date_created', ascending=False)

        
        # Prioritize articles based on user preferences for categories
        if user_preferences:
            category_weights = user_preferences
            category_max_score = max(category_weights.values(), key=lambda x: x['score'])['score']
            for category, weight_info in category_weights.items():
                weight = weight_info['score']
                self.data.loc[self.data['category'] == category, 'combined_similarity'] *= (weight / category_max_score)
        
        # Sort again by adjusted combined similarity
        self.data = self.data.sort_values(by='combined_similarity', ascending=False)
        
        return self.data



