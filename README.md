# Customer Segmentation using ML.NET

## Overview

This project builds a customer segmentation system using ML.NET to group users based on behavioral patterns derived from book rating data.

Instead of treating users as a single population, the system identifies distinct user segments such as:

* Enthusiast readers
* Casual users
* Harsh critics
* Generous raters

The goal is to enable data driven personalization and targeted strategies.

---

## Problem

Raw user data does not provide actionable insights.

Challenges:

* Large number of users with different behaviors
* Highly inconsistent rating patterns
* No clear grouping of users for personalization

Without segmentation:

* recommendations become generic
* engagement strategies are ineffective

---

## Solution

The system builds user profiles and applies clustering to group similar users.

Pipeline:

Raw Ratings Data
→ User Profile Engineering
→ Feature Normalization
→ KMeans Clustering
→ Segment Analysis

---

## Feature Engineering

Each user is transformed into a behavioral profile using:

* TotalRatings
* AverageRating
* PercentLiked (ratings >= 8)
* PercentDisliked (ratings <= 4)
* RatingVariance

These features capture both activity and sentiment behavior.

---

## Normalization

All features are scaled to 0 to 1 range to ensure:

* no feature dominates clustering
* fair distance calculation

---

## Model Selection (Elbow Method)

The system automatically determines the optimal number of clusters.

Process:

* Train KMeans for K = 2 to 10
* Compute inertia (distance to cluster center)
* Select K where improvement drops below threshold

This avoids arbitrary cluster selection.

---

## Model

* Algorithm: KMeans Clustering
* Framework: ML.NET
* Distance metric: Euclidean

---

## Output

Each user is assigned a segment based on behavior.

Example segments:

* Enthusiast Readers
  High activity and high ratings

* Casual Fans
  Low activity but positive ratings

* Harsh Critics
  Low average ratings

* Generous Raters
  High percentage of positive ratings

---

## Sample Output

```id="t4v1zv"
Segment 1 — Enthusiast Readers
Users: 120
Avg Ratings: 45
Avg Score: 8.6

Segment 2 — Harsh Critics
Users: 80
Avg Ratings: 30
Avg Score: 4.2
```

---

## Tech Stack

* .NET
* ML.NET
* C#

---

## How to Run

1. Add dataset paths in code
2. Run the project

```bash id="mj2qk2"
dotnet run
```

---

## Key Learnings

* Feature engineering is more important than the algorithm
* Data distribution heavily impacts clustering quality
* Automatic K selection avoids arbitrary decisions
* Behavioral interpretation is critical after clustering

---

## Limitations

* No real time updates
* Dataset is static
* No visualization layer

---

## Future Improvements

* Add visualization for clusters
* Integrate with recommendation system
* Add real time segmentation pipeline

---

## Why This Project Matters

This is not just a clustering demo.

It demonstrates:

* Translating raw data into meaningful behavioral features
* Applying unsupervised learning in a real use case
* Converting clusters into actionable user segments

---
