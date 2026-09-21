import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


print("Loading dataset...")
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
print(f"Dataset Loaded Successfully! Total items: {len(df)}")


print("Converting text overviews into numbers (TF-IDF)...")
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])


print("Calculating similarity matrix...")
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend(movie_title, top_n=2):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    
    if matches.empty:
        return f"Movie '{movie_title}' not found in dataset."
        
    idx = matches.index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    recommended_indices = [item[0] for item in sorted_scores[1:top_n+1]]
    return df[['title', 'overview']].iloc[recommended_indices]

print("\n--- Recommendations for 'Avatar' ---")
print(recommend('Avatar'))

print("\n--- Recommendations for 'The Dark Knight Rises' ---")
print(recommend('The Dark Knight Rises'))
