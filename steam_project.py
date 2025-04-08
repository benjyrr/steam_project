# --- 1. Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 2. Load Datasets ---
reviews_df = pd.read_csv("steam_tags_project/archive/steam.csv")
tags_df = pd.read_csv("steam_tags_project/archive/steamspy_tag_data.csv")

# --- 3. Create Proxy Review Score ---
reviews_df['total_ratings'] = reviews_df['positive_ratings'] + reviews_df['negative_ratings']
reviews_df = reviews_df[reviews_df['total_ratings'] > 0]
reviews_df['review_score'] = reviews_df['positive_ratings'] / reviews_df['total_ratings']

# --- 4. Filter Popular Games ---
# Keep only games that appear in both datasets
valid_appids = set(reviews_df['appid']) & set(tags_df['appid'])
reviews_df = reviews_df[reviews_df['appid'].isin(valid_appids)]

# Now filter for popular games
reviews_df['total_ratings'] = reviews_df['positive_ratings'] + reviews_df['negative_ratings']
reviews_df = reviews_df[reviews_df['total_ratings'] > 1000]
reviews_df['review_score'] = reviews_df['positive_ratings'] / reviews_df['total_ratings']

# --- 5. Merge with Tag Data ---

# Check if top-rated games exist in tag dataset before merging
pre_merge_top_reviewed = reviews_df.groupby('name')['review_score'].mean().sort_values(ascending=False).head(10)
print("Top rated games overall (before merge):")
print(pre_merge_top_reviewed.index.tolist())
top_appids = reviews_df[reviews_df['name'].isin(pre_merge_top_reviewed.index)]['appid'].unique()
print("Are these AppIDs in tag dataset?")
print(set(top_appids) & set(tags_df['appid'].unique()))

merged_df = pd.merge(reviews_df, tags_df, on='appid', how='left')

tag_columns = [col for col in tags_df.columns if col != 'appid']

# --- 6. Top-Rated Games ---
avg_scores = merged_df.groupby('name')['review_score'].mean()
top_loved_games = avg_scores.sort_values(ascending=False).head(10)

# --- 7. Most Polarizing Games ---
score_std = merged_df.groupby('name')['review_score'].std()
most_polarizing = score_std.sort_values(ascending=False).head(10)

# --- 8. Tag Frequency Among Top-Rated Games ---
top_games = top_loved_games.index.tolist()
top_df = merged_df[merged_df['name'].isin(top_games)]

# Debug print to verify tag data
print("Top-rated game names:", top_games)
if not top_df.empty:
    print("Tag sum for top games:", top_df[tag_columns].sum().sort_values(ascending=False).head(10))
    print("Sample top_df rows:", top_df[tag_columns].head())
else:
    print("Warning: top_df is empty. Check if tag merge failed or top_loved_games has no matches.")
top_games = top_loved_games.index.tolist()
top_df = merged_df[merged_df['name'].isin(top_games)]

top_tags_sum = top_df[tag_columns].sum().sort_values(ascending=False).head(30)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_tags_sum.values, y=top_tags_sum.index, palette='rocket')
plt.title('Most Common Tags in Top-Rated Steam Games (SteamSpy Data)')
plt.xlabel('Number of Games Tagged')
plt.ylabel('Tag')
plt.tight_layout()
plt.savefig("top_rated_game_tags_binary.png")
plt.show()

# --- 9. Logistic Regression Model ---
# Binarize review_score for classification
# 1 = liked (>= 0.75), 0 = not liked
y = (merged_df['review_score'] >= 0.75).astype(int)
X = merged_df[tag_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
if len(X_train) == 0:
    print("Error: Not enough data for training. Check if tag columns are all zero or filtered too strictly.")
    exit()
model.fit(X_train, y_train)

# --- 10. Evaluation ---
print("Top predictive tags (coefficients):")
coeffs = pd.Series(model.coef_[0], index=tag_columns)
print(coeffs.sort_values(ascending=False).head(10))
print("Least predictive tags:")
print(coeffs.sort_values().head(10))
print("Model Accuracy:", model.score(X_test, y_test))

# --- 11. K-Means Clustering on Review Scores + Tags ---

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

# Select features: review_score, total_ratings, and tag columns
cluster_features = merged_df[['review_score', 'total_ratings'] + tag_columns].dropna()

# Scale for K-Means
scaler = StandardScaler()
scaled_features = scaler.fit_transform(cluster_features)

# Run K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
merged_df['cluster'] = kmeans.fit_predict(scaled_features)

# Reduce dimensions for plotting
pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)

# Add PCA back to DataFrame
merged_df['PCA1'] = pca_components[:, 0]
merged_df['PCA2'] = pca_components[:, 1]

# Plot clusters
plt.figure(figsize=(10, 6))
sns.set(style="whitegrid")
palette = sns.color_palette("Set2", n_colors=merged_df['cluster'].nunique())

scatter = sns.scatterplot(
    data=merged_df,
    x='PCA1',
    y='PCA2',
    hue='cluster',
    palette=palette,
    alpha=0.7,
    edgecolor='black',
    s=60
)

plt.title('K-Means Clustering of Steam Games', fontsize=16)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("steam_kmeans_clusters_clean.png", dpi=300)
plt.show()

# Optional: View cluster summary
cluster_summary = merged_df.groupby('cluster')[['review_score', 'total_ratings'] + tag_columns].mean()
print("Cluster summary (average values per cluster):")
print(cluster_summary.sort_values('review_score', ascending=False))

# --- 12. Cluster Analysis ---

# Show number of games in each cluster
print("Number of games per cluster:")
print(merged_df['cluster'].value_counts())

# Display 5 sample games from each cluster
for cluster_id in sorted(merged_df['cluster'].unique()):
    print(f"\n🎮 Cluster {cluster_id} sample games:")
    sample_games = merged_df[merged_df['cluster'] == cluster_id][['name', 'review_score', 'total_ratings']]
    if not sample_games.empty:
        print(sample_games.sample(min(5, len(sample_games))).to_string(index=False))
    else:
        print("No games in this cluster.")

# Display top 10 tags per cluster
for cluster_id in sorted(merged_df['cluster'].unique()):
    print(f"\n🏷️ Top tags in Cluster {cluster_id}:")
    tag_means = merged_df[merged_df['cluster'] == cluster_id][tag_columns].mean()
    top_tags = tag_means.sort_values(ascending=False).head(10)
    print(top_tags.to_string())