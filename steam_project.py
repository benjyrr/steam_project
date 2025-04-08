# --- 1. Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter

# --- 2. Load Datasets ---
reviews_df = pd.read_csv("archive/steam.csv")
tags_df = pd.read_csv("archive/steamspy_tag_data.csv")

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
