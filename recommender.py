import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
data = {
    'title': [
        'Avatar',
        'John Carter',
        'The Dark Knight Rises',
        'Spectre',
        'Spider-Man 3'
    ],
    'overview': [
        'In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission.',
        'John Carter is a war-weary, former military captain who is inexplicably transported to Mars.',
        'Following the death of District Attorney Harvey Dent, Batman assumes responsibility for Dent\'s crimes.',
        'A cryptic message from the past sends James Bond on a rogue mission to Mexico City and Rome.',
        'A strange black entity from another world bonds with Peter Parker and causes inner turmoil.'
    ]
}

df = pd.DataFrame(data)


tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, sim_matrix=cosine_sim, df=df):
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices[title]
    
    sim_scores = list(enumerate(sim_matrix[idx]))
  
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
  
    sim_scores = sim_scores[1:]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [i[1] for i in sim_scores]
    
    result_df = df.iloc[movie_indices].copy()
    result_df['similarity_score'] = scores
    return result_df

target_movie = 'Avatar'
recommendations = get_recommendations(target_movie)
print(f"--- Recommendations for '{target_movie}' ---")
print(recommendations[['title', 'similarity_score']])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(
    cosine_sim, 
    annot=True, 
    fmt=".2f", 
    cmap="YlGnBu", 
    xticklabels=df['title'], 
    yticklabels=df['title'],
    ax=axes[0]
)
axes[0].set_title("Cosine Similarity Heatmap", fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)

sns.barplot(
    x='similarity_score', 
    y='title', 
    data=recommendations, 
    palette="Blues_r", 
    ax=axes[1]
)
axes[1].set_title(f"Top Recommendations for '{target_movie}'", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Similarity Score")
axes[1].set_ylabel("Movie Title")

plt.tight_layout()

plt.savefig("output.png", dpi=300)
print("\n[SUCCESS] Heatmap & Recommendation Bar Chart saved as 'output.png'.")

plt.show()
