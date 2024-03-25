import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_json('data/bloomberg_quint_news.json')

vectorizer = TfidfVectorizer(stop_words='english')
df['short_description'] = df['short_description'].fillna('')
X_short_desc = vectorizer.fit_transform(df['short_description'])

category_vectorizer = TfidfVectorizer()
category_matrix = category_vectorizer.fit_transform(df['category'])

cosine_sim_short_desc = linear_kernel(X_short_desc, X_short_desc)
category_cosine_sim = linear_kernel(category_matrix, category_matrix)

titles = pd.Series(df.index, index=df['title']).drop_duplicates()

user_preferences = {
    'Politics': 4,
    'Technology': 3,
    'Business': 5,
    'Sports': 2,
    'Entertainment': 4
}


def get_recommendation_based_on_user_preferences_scores(title, cosine_sim=cosine_sim_short_desc, category_cosine_sim=category_cosine_sim,
                        user_preferences=user_preferences, weight=0.5):
    idx = titles[title]

    category_idx = df.iloc[idx]['category']
    category_weight = user_preferences.get(category_idx, 0) / 5.0  # Normalize to range [0, 1]

    combined_scores = [(i, weight * cosine_sim[idx][i] + (1 - weight) * category_weight * category_cosine_sim[idx][i])
                       for i in range(len(cosine_sim))]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    news_indices = [i[0] for i in combined_scores[1:11]]

    return df['title'].iloc[news_indices]


#print(get_recommendation_based_on_user_preferences_scores('India’s Covid-19 Cases Top 20 Million as Infections Continue to Spread'))


def get_recommendation_based_on_keyword_search_in_the_title(keyword, cosine_sim=cosine_sim_short_desc):
    # search titles that contain the keyword
    indices = [i for i in range(len(titles)) if keyword.lower() in titles.index[i].lower()]

    combined_scores = [(i, cosine_sim[indices[0]][i]) for i in indices]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)
    
    news_indices = [i[0] for i in combined_scores[1:11]]
    
    return df['title'].iloc[news_indices]


def get_recommendation_based_on_keyword_search_in_the_short_description(keyword, cosine_sim=cosine_sim_short_desc, recent = False):
    # search titles that contain the keyword
    indices = [i for i in range(len(titles)) if keyword.lower() in df['short_description'].iloc[i].lower()]
    
    if recent:
        # convert to date time
        df['date_created'] = pd.to_datetime(df['date_created'])
        print(df['date_created'])
        # get the latest news
        latest_news = df['date_created'].idxmax()
        indices = [i for i in indices if i == latest_news]

    combined_scores = [(i, cosine_sim[indices[0]][i]) for i in indices]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    news_indices = [i[0] for i in combined_scores[1:11]]

    return df['title'].iloc[news_indices]






print(get_recommendation_based_on_keyword_search_in_the_short_description('covid', recent=True))