**Project Title: Predicting Game Ratings from Tags — A Data Science Case Study**

**Overview**
In this project, I apply end-to-end data science techniques to understand which gameplay features drive high user satisfaction on Steam. Using over 6 million user reviews and 370+ gameplay tags from SteamSpy, I built a predictive model that identifies which tags are most associated with highly rated games. The final logistic regression model achieved 74% accuracy and provides interpretable business insights for game studios, product teams, and platform curators.

**Objective**
To identify which gameplay tags are most predictive of high user ratings on Steam, and quantify their importance using machine learning.

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

**Modeling Approach**
- Trained a logistic regression classifier using tag features to predict well-rated status
- Achieved **74% accuracy** on test data (20% holdout split)
- Extracted model coefficients to assess tag influence

**Top Predictive Tags (Positive Influence)**
- `touch_friendly`, `soundtrack`, `runner`, `minimalist`, `score_attack`
- Suggest accessible, tightly-scoped experiences perform well

**Least Predictive Tags (Negative Influence)**
- `jet`, `tutorial`, `snowboarding`, `gamemaker`, `otome`
- May indicate complexity, dated engines, or niche appeal

**Key Takeaways**
- Binary tag data can meaningfully predict user sentiment at scale
- Certain tags (e.g., `touch_friendly`) have clear positive associations across genres
- Results could guide product managers, UX researchers, or recommender systems in prioritizing features and positioning games

**Next Steps**
- Try tree-based models for higher accuracy (e.g., random forest, XGBoost)
- Incorporate NLP from review text to capture qualitative sentiment
- Build a Streamlit or Dash app for interactive tag exploration

---
*Created as part of my portfolio exploring behavioral data, product feedback, and predictive modeling in games and technology.*

