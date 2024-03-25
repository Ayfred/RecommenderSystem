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

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

user_preferences = {
    'Politics': 4,
    'Technology': 3,
    'Business': 5,
    'Sports': 2,
    'Entertainment': 4
}


def get_recommendations(title, cosine_sim=cosine_sim_short_desc, category_cosine_sim=category_cosine_sim,
                        user_preferences=user_preferences, weight=0.5):
    idx = indices[title]

    category_idx = df.iloc[idx]['category']
    category_weight = user_preferences.get(category_idx, 0) / 5.0  # Normalize to range [0, 1]

    combined_scores = [(i, weight * cosine_sim[idx][i] + (1 - weight) * category_weight * category_cosine_sim[idx][i])
                       for i in range(len(cosine_sim))]
    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    news_indices = [i[0] for i in combined_scores[1:11]]

    return df['title'].iloc[news_indices]


print(get_recommendations('All You Need To Know Going Into Trade On September 23'))
