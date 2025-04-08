**Project Title: Predicting & Clustering Game Ratings from Tags — A Data Science Case Study**

**Overview**  
In this project, I apply end-to-end data science techniques to understand which gameplay features drive high user satisfaction on Steam — and how games can be clustered by review behavior and genre. Using over 6 million user reviews and 370+ gameplay tags from SteamSpy, I built both a predictive model and an unsupervised clustering model to surface patterns in game design, audience reception, and genre structure. The logistic regression model achieved 74% accuracy, while K-Means clustering revealed clear segments including mainstream core games, educational sims, competitive FPS titles, and hardcore strategy games.

**Objective**  
To identify which gameplay tags are most predictive of high user ratings on Steam, and to uncover natural groupings of games based on genre and popularity using unsupervised learning.

**Data Sources**  
- **Steam reviews dataset**: Includes game ID, name, and counts of positive/negative user ratings  
- **SteamSpy tag dataset**: Binary indicators for 370+ gameplay tags (e.g., "open_world", "puzzle")

**Data Engineering & Feature Prep**  
- Merged datasets on `appid`, retaining only games present in both  
- Filtered for popular titles (≥1,000 total reviews) to ensure stable rating estimates  
- Computed proxy review score: `positive_ratings / (positive + negative)`  
- Binarized target variable: 1 = well-rated (score ≥ 0.75), 0 otherwise  
- Extracted 370+ binary features representing tags

**Exploratory Data Analysis (EDA)**  
- Identified top-rated games with broad user bases (e.g., *Portal 2*, *Baba Is You*)  
- Created bar chart of most common tags among top-rated games  
- Observed dominance of *puzzle*, *co_op*, and *singleplayer* tags  
- Identified the most polarizing games using the standard deviation of review scores

**Modeling Approach**

- **Logistic Regression (Supervised)**  
  Trained a classifier using tag features to predict well-rated status  
  - Achieved **74% accuracy** on test data (20% holdout split)  
  - Extracted model coefficients to assess tag influence  

  **Top Predictive Tags (Positive Influence)**  
  - `touch_friendly`, `soundtrack`, `runner`, `minimalist`, `score_attack`  
  - Suggest accessible, tightly-scoped experiences perform well

  **Least Predictive Tags (Negative Influence)**  
  - `jet`, `tutorial`, `snowboarding`, `gamemaker`, `otome`  
  - May indicate complexity, dated engines, or niche appeal

- **K-Means Clustering (Unsupervised)**  
  Applied K-Means clustering (k=4) to standardized review scores, review volume, and tag features  
  - Reduced dimensions with PCA for 2D visualization  
  - Discovered four natural segments in the Steam ecosystem:

    **Cluster 0 – Mainstream Core Games**  
    High-volume action, RPG, and multiplayer titles with broad genre coverage  
    *(e.g., Left 4 Dead 2, Skullgirls)*  

    **Cluster 1 – Niche Sim/Logic Games**  
    Smaller-scale educational or programming-focused games  
    *(e.g., while True: learn())*

    **Cluster 2 – Multiplayer FPS Juggernaut**  
    Dominated by *CS:GO*, separated by scale and tag intensity  
    *(e.g., Counter-Strike: Global Offensive)*  

    **Cluster 3 – Hardcore Strategy Enthusiasts**  
    Deep strategy titles with long playtimes and niche historical content  
    *(e.g., Civ V, Europa Universalis IV, FTL)*  

    > Note: Some clusters contained only one game due to extreme outlier behavior. Rather than discard these points, they were retained to highlight the scale and specialization of unique titles like *CS:GO* and educational sims.

**Key Takeaways**  
- Binary tag data can meaningfully predict user sentiment at scale  
- Certain tags (e.g., `touch_friendly`) have strong positive associations across genres  
- Clustering reveals hidden genre structures and outlier dominance  
- Combined modeling and clustering can guide product managers, UX researchers, and recommender systems in game discovery and design analysis

**Next Steps**  
- Explore alternative clustering techniques (e.g., DBSCAN, HDBSCAN) to better handle outliers  
- Try tree-based models (e.g., Random Forest, XGBoost) for improved prediction  
- Add interactive visualizations or dashboards (e.g., Streamlit or Dash)  
- Incorporate NLP on review text to extract deeper sentiment patterns

---
*Created as part of my portfolio exploring behavioral data, product feedback, and machine learning in games and digital experiences.*
