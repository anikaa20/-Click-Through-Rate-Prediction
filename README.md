# Maximizing Digital Advertising Revenue Through Enhanced User Engagement Analytics 



## Project Story 

<p align="center">

<!-- Deck GIF here -->

<img src="https://github.com/anikaa20/-Click-Through-Rate-Prediction/blob/main/deck_gif.gif" width="900"/>
</p>
<p align="center">
  <a href="https://docs.google.com/presentation/d/1ihN5al63xyfBj0pz7i_CafzmVrCMuI26/edit?usp=sharing&ouid=108825657904597871730&rtpof=true&sd=true">
    📊 <b>View the Complete Project Deck</b>
  </a>
</p>

## Project Background

Digital advertising operates on a simple but costly assumption: knowing *who* a user is tells the organization whether they will engage. This project challenges that assumption head-on, **does user engagement follow demographic patterns, or is it governed by something more complex?**

Analysis is based on a dataset of 10,000 user records sourced from Kaggle, spanning key variables across demographics, geography, and behavioral signals: `Age`, `Area Income`, `Daily Time Spent on Site`, `Daily Internet Usage`, `Gender`, `Country`, and `Clicked on Ad`.

Insights and recommendations are organized across four areas:

- **Behavioral Patterns**: The bimodal engagement split and what it reveals about user archetypes
- **Demographic Analysis**: Quantifying the role of age, income, and gender in click behavior
- **Geographic Signals**: Regional CTR variance and its implications for campaign strategy
- **Model Progression**: Why complexity wins and how XGBoost outperforms simpler classifiers

Full Python EDA code → [`EDA_CTR.ipynb`](https://github.com/anikaa20/-Click-Through-Rate-Prediction/blob/main/EDA_CTR.ipynb)  
Full ML pipeline → [`CTR.ipynb`](https://github.com/anikaa20/-Click-Through-Rate-Prediction/blob/main/CTR.ipynb)

---

## Data Structure & Initial Checks

**Source:** User Advertisement Interaction Dataset (Kaggle)

**Raw dataset:** 10,000 rows × 9 columns → **Cleaned dataset:** ~9,000 rows × 8 columns

| Cleaning Step | Action |
|---|---|
| Column removal | Dropped unnamed index artifact column |
| Deduplication | Removed duplicate rows |
| Missing value check | Confirmed no null values via `.info()` |
| Target balance check | 49.17% CTR; confirmed balanced, no resampling needed |
| Categorical encoding | Target encoding applied to `Country`, `City`, `Gender` |
| Feature scaling | `StandardScaler` applied to continuous numerical columns |

---

## Executive Summary

User engagement is not a demographics problem; it is a **complexity problem**. EDA revealed weak linear correlations across all features, which initially appears contradictory given observable engagement patterns. The explanation: CTR is driven by **interactions between variables**, not individual features in isolation. The effect of age shifts across regions; time-on-site behaves differently across demographic segments. This non-linearity renders traditional rule-based targeting insufficient.

Four models were evaluated to test this hypothesis. Results validated it unambiguously: **as model sophistication increased, performance improved consistently**, with XGBoost delivering the strongest outcome at **88% precision, recall, and F1 score**. The gap between Logistic Regression (0.71 F1) and XGBoost (0.88 F1) is not a minor algorithmic edge, it is the measurable cost of assuming linearity, where none exists.

---

## Insights Deep Dive

### Category 1: Behavioral Patterns

* `Daily Time Spent on Site` emerged as the **primary driver of engagement**, exhibiting a distinct bimodal distribution across the dataset.
* Two clear user archetypes exist: a **High-Engagement Core** (80+ minutes on-site, low click propensity likely brand browsers) and **Low-Engagement Casuals** (under 45 minutes, higher click rate likely intent-driven visitors).
* This inversion, less time on site correlating with higher click rates is counter-intuitive but consistent, suggesting short-session users arrive with a specific purpose and are more receptive to direct ad conversion.


### Category 2: Demographic Analysis

* Peak engagement age is concentrated around **36 years**, with an average area income of **$53,840**, pointing to established mid-career professionals as the highest-engagement cohort.
* Females represent **53.9% of users** but account for **56.8% of clicks** (~3% response uplift), suggesting a mild but not dominant gender signal.
* Area income shows a unimodal distribution peaking at $60,000. While purchasing power is a stable covariate, it shows weak linear correlation with click behavior, reinforcing that income alone is not a reliable predictor.

### Category 3: Geographic Signals

* Geographic region introduces measurable CTR variance: **Asia outperforms other regions**, while **Europe underperforms**.
* Country-level data shows wide dispersion, with the top 10 countries by click share collectively representing a concentrated engagement base.
* Regional CTR differences are actionable; budget allocation, creative localization, and campaign intensity should be tiered by regional performance rather than applied uniformly.

### Category 4: Model Progression

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression | 0.72 | 0.71 | 0.71 |
| Decision Tree | 0.78 | 0.77 | 0.77 |
| Random Forest | 0.85 | 0.85 | 0.85 |
| **XGBoost** | **0.88** | **0.88** | **0.88** |

* The performance progression is not coincidental, it is structural. Each step up in model complexity corresponds to an improved ability to capture **non-linear feature interactions**, which is precisely what drives CTR.
* XGBoost was tuned via `GridSearchCV` with 5-fold cross-validation across `n_estimators`, `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree`.
* **XGBoost's advantage is not algorithmic luck, it is evidence that the underlying customer behavior itself is non-linear.**

---

## Recommendations

* **Shift from demographic segmentation to behavioral scoring:** Replace static audience rules with a predictive scoring layer powered by XGBoost. Target users based on behavioral signal combinations and not demographic buckets.

* **Prioritize the Low-Engagement Casual archetype for direct-response campaigns:** Users with under 45 minutes on-site show higher click propensity. Creative and placement strategy for this segment should be optimized for conversion.

* **Allocate disproportionate budget toward Asia, re-evaluate European strategy:** Regional CTR data is sufficiently divergent to warrant differentiated campaign architecture and not just localized copy.

* **Build an audience scoring pipeline using the XGBoost model:** Integrate real-time behavioral inputs (time-on-site, internet usage patterns) to score incoming users before ad delivery, enabling proactive rather than reactive targeting.

* **Treat age 30–40 as the refinement zone, not the core target:** Significant overlap between clickers and non-clickers in this band means blanket targeting here is wasteful. Model-based segmentation, not age cutoffs should govern this cohort.

---

## Assumptions & Caveats

* **Target encoding applied to high-cardinality features** (`Country`, `City`) captures geographic signal without dimensionality explosion, but introduces mild data leakage risk in production; a holdout encoding scheme is advisable at deployment.
* **Dataset represents a single observational snapshot**: Temporal drift in engagement behavior is not captured. Model retraining cadence should be established before production deployment.
* **Balanced dataset (49.17% CTR) is atypical for real-world ad data** where CTR is usually well below 5%. Findings should be validated against a production-scale, imbalanced dataset before drawing conclusions about real campaign performance.
* **Outliers were retained** in numerical features, as they reflect genuine behavioral variance (rare but real high-usage events) rather than data error — removing them would have suppressed valid signal.
* **XGBoost hyperparameter tuning was computationally intensive** (large grid across 5 parameters × 5-fold CV); results reflect the best configuration found within the defined search space, not a global optimum.
