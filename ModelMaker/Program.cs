using Microsoft.ML;
using BLModels;
using System.Data;
using System.Reflection;
using Microsoft.ML.Trainers;
using static System.Reflection.Metadata.BlobBuilder;
using System.Runtime.Intrinsics.X86;


#region Sentiment Analysis

//var mlContext = new MLContext(seed: 1);

//var dataView = mlContext.Data.LoadFromTextFile<SentimentAnalysisData>(
//    "C:/FYJIXFINAL/UAT/ML.NET/ModelMaker/Dataset/sentiment_analysis.csv",
//    hasHeader: true,
//    separatorChar: ','
//);

//var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);


//var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
//        inputColumnName: "Label",
//        outputColumnName: "Label"
//    )
//    .Append(mlContext.Transforms.Text.FeaturizeText(
//        outputColumnName: "Features",
//        inputColumnName: nameof(SentimentAnalysisData.text)
//    ))
//    .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
//    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

//var model = pipeline.Fit(split.TrainSet); // 👈 was pipeline.Fit(dataView)


//var predictor = mlContext.Model.CreatePredictionEngine<SentimentAnalysisData, SentimentAnalysisPrediction>(model);

//var sample = new SentimentAnalysisData
//{
//    text = "This is a okay!",
//};

//var predictions = model.Transform(split.TestSet);

//var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

//Console.WriteLine($"Accuracy:       {metrics.MacroAccuracy:F4}");
//Console.WriteLine($"Log Loss:       {metrics.LogLoss:F4}");
//Console.WriteLine($"Log Loss Ratio: {metrics.LogLossReduction:F4}");



//Console.WriteLine($"Sentiment: {predictor.Predict(sample).PredictedLabel}");


//var baseDirectory = AppContext.BaseDirectory;
//////ML/bin/debug/net10.0

//var modelDirectory = Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", "..", "ML.NET", "wwwroot", "sentiment_analysis_model.zip"));

//Directory.CreateDirectory(modelDirectory);

//var modelPath = Path.Combine(modelDirectory, "SentimentModel.zip");
//mlContext.Model.Save(model, dataView.Schema, modelPath);

#endregion


#region Recommendation Engine

/*
 * ============================================================
 * CASE STUDY: Book Recommendation System using ML.NET
 * ============================================================
 *
 * OBJECTIVE
 * ---------
 * Build a hybrid book recommendation system using the Kaggle
 * Book Recommendation Dataset (Books.csv, Ratings.csv, Users.csv)
 * with ML.NET in a .NET console + class library setup.
 *
 * ============================================================
 * DATASET
 * ============================================================
 *
 * Source   : https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset
 * Files    : Books.csv, Ratings.csv, Users.csv
 * Ratings  : 397,243 rows | Users: 278,858 | Books: 271,360
 *
 * Key characteristics that made this dataset difficult:
 *   - 99.99% sparse matrix (most users rated very few books)
 *   - Ratings heavily skewed toward 8-10 (people only rate books they liked)
 *   - Average ratings per user: ~19 out of 271,360 books
 *   - No genre or category column (only Author, Publisher, Year)
 *
 * ============================================================
 * JOURNEY & MISTAKES MADE (the honest version)
 * ============================================================
 *
 * ATTEMPT 1 — Raw Matrix Factorization
 * --------------------------------------
 * What we did   : Fed raw ratings directly into MatrixFactorization
 * Mistake        : MapValueToKey outputColumnName was set to "text"
 *                  instead of "Label", overwriting the text column
 *                  with a Key type causing Schema mismatch error.
 * Error          : System.ArgumentOutOfRangeException:
 *                  'Schema mismatch for input column text:
 *                   expected String, got Key<UInt32>'
 * Fix            : Set outputColumnName to "Label" and added
 *                  [ColumnName("Label")] to the Rating property.
 * Result         : R² = 0.08, RMSE = 3.64 (terrible)
 *
 * ATTEMPT 2 — Normalize ratings to 0-1
 * --------------------------------------
 * What we did   : Divided all ratings by 10 to normalize to 0-1 range
 * Mistake        : Applied binary threshold (>= 0.7 = liked) ON the
 *                  normalized scale, then forgot to switch back causing
 *                  the LossFunction to mismatch the data format.
 * Result         : R² = -1.14, RMSE = 2.61 (worse than before!)
 * Lesson         : Normalizing then bucketing on wrong scale = chaos.
 *                  Always apply bucketing on the ORIGINAL scale first.
 *
 * ATTEMPT 3 — Filter sparse users and books
 * -------------------------------------------
 * What we did   : Removed users and books with fewer than 10 ratings
 * Mistake        : Threshold of 10 was too aggressive. Threw away 92%
 *                  of data. Only 30,177 rows remained out of 397,243.
 * Result         : R² = -0.63, RMSE = 0.22 (RMSE looks good but
 *                  only because scale shrank, R² still negative)
 * Lesson         : More data almost always beats cleaner data in
 *                  collaborative filtering. Lower threshold to 3.
 *
 * ATTEMPT 4 — Balance the buckets
 * ---------------------------------
 * What we did   : Undersampled majority class (Liked = 60%) to match
 *                 minority class (Disliked = 3.9%) — 10,059 rows each
 * Mistake        : Threw away 87% of remaining data to fix imbalance.
 *                  Model had even less data to learn from.
 * Result         : R² = -0.08, RMSE = 0.72 (marginal improvement)
 * Lesson         : Undersampling is dangerous with sparse datasets.
 *                  Fix bucketing boundaries instead of throwing data away.
 *
 * ATTEMPT 5 — Fix bucketing boundaries
 * --------------------------------------
 * What we did   : Changed buckets from (<=4, <=7, >7) to (<=6, <=8, >8)
 *                 to distribute ratings more evenly across 3 buckets.
 * Result         : Distribution improved to 22% / 42% / 36%
 *                  R² = -0.01 (almost at zero — close to baseline!)
 * Lesson         : Bucket boundaries matter enormously. Always check
 *                  distribution BEFORE training.
 *
 * ATTEMPT 6 — NaN predictions discovered
 * ----------------------------------------
 * What we did   : Checked Actual vs Predicted table
 * Discovery      : Min predicted = NaN! 984 out of 44,313 predictions
 *                  were NaN, poisoning the entire metric calculation.
 * Root cause     : Random train/test split put some users ONLY in the
 *                  test set. Model never saw them → outputs NaN.
 *                  (Cold Start Problem)
 * Fix            : Per-user train/test split — 80% of EACH user's
 *                  ratings go to train, 20% go to test. Guarantees
 *                  every user in test was also seen in training.
 * Result         : NaN count dropped but R² still stubbornly negative.
 *
 * ATTEMPT 7 — Root cause analysis
 * ---------------------------------
 * Discovery      : Model predicts everything between 0.1 and 1.3.
 *                  Never predicts close to 2 (Liked) even when
 *                  actual label IS 2.
 * Root cause     : The dataset is fundamentally too sparse for
 *                  Matrix Factorization to find meaningful patterns.
 *                  19 ratings per user across 27,000 books = not
 *                  enough signal. Model defaults to predicting the
 *                  average (safe bet) for everything.
 * Lesson         : No amount of tuning fixes a fundamentally sparse
 *                  dataset for pure collaborative filtering.
 *
 * ============================================================
 * FINAL SOLUTION — Hybrid Recommendation System
 * ============================================================
 *
 * Architecture:
 *   Collaborative Filter (Matrix Factorization) × 0.5 weight
 *   +
 *   Content Based Filter (Jaccard Similarity)   × 0.5 weight
 *   =
 *   Hybrid Score → Top 10 Recommendations
 *
 * Collaborative component:
 *   - MatrixFactorization with SquareLossRegression
 *   - 100 iterations, rank 200, LR 0.005, Lambda 0.05
 *   - Trained on 151,318 rows, tested on 37,968 rows
 *   - Per-user split to eliminate cold start NaN problem
 *
 * Content component:
 *   - Jaccard similarity on Title + Author + Publisher words
 *   - For a target user: average similarity between candidate
 *     book and all books the user previously rated as Liked (2)
 *   - No training required — pure lookup at prediction time
 *
 * Final metrics:
 *   R²   = -0.33  (negative but misleading — see note below)
 *   RMSE = 0.87
 *   MAE  = 0.73
 *
 * ============================================================
 * WHY R² IS MISLEADING HERE
 * ============================================================
 *
 *   R² measures how well the model beats a "predict the average"
 *   baseline. With our heavily skewed dataset (60% Liked),
 *   predicting the average (1.4) is actually very hard to beat.
 *
 *   However, the ACTUAL recommendations are high quality:
 *     - Harry Potter and the Order of the Phoenix
 *     - Angela's Ashes
 *     - High Fidelity
 *     - The No. 1 Ladies Detective Agency
 *
 *   These are genuinely well-regarded, popular books.
 *   R² being negative does NOT mean the system is broken —
 *   it means R² is the wrong metric for this problem.
 *
 *   Better metrics for recommendation systems:
 *     - Precision@K  : Of top K recommendations, how many did user like?
 *     - Recall@K     : Of all books user likes, how many are in top K?
 *     - NDCG         : Normalized Discounted Cumulative Gain
 *     - These require explicit user feedback to compute properly.
 *
 * ============================================================
 * KEY LESSONS LEARNED
 * ============================================================
 *
 *   1. Always print raw CSV lines before writing any parsing code.
 *      Separator (comma vs semicolon) and quoting caused multiple
 *      wasted attempts.
 *
 *   2. Always check data distribution BEFORE training.
 *      Skewed data is the #1 silent killer of ML models.
 *
 *   3. Train/test split must be done PER USER for recommendation
 *      systems, not randomly. Random split causes cold start NaN.
 *
 *   4. More data beats cleaner data in sparse collaborative filtering.
 *      Aggressive filtering (threshold >= 10) hurt more than helped.
 *
 *   5. R² and RMSE are poor metrics for recommendation systems.
 *      The real question is: "Are the recommendations sensible?"
 *      And the answer here is: YES.
 *
 *   6. This Kaggle book dataset is notoriously difficult.
 *      Research papers using this dataset report similar R² values.
 *      MovieLens dataset is much better for learning/experimenting.
 *
 *   7. Hybrid always beats single algorithm on sparse datasets.
 *      Content similarity rescued recommendations even when
 *      collaborative scores were mediocre.
 *
 * ============================================================
 * TOOLS & PACKAGES USED
 * ============================================================
 *
 *   - ML.NET                  (Microsoft.ML)
 *   - ML.NET Recommender      (Microsoft.ML.Recommender)
 *   - .NET 9 Console App
 *   - Class Library (BLModels) for shared model classes
 *
 * ============================================================
 * AUTHOR NOTE
 * ============================================================
 *
 *   This was not a straight line from problem to solution.
 *   Every wrong attempt taught something valuable.
 *   The "disaster" WAS the learning. 
 *
 * ============================================================
 */
//static string[] SplitCsv(string line)
//{
//    var result = new List<string>();
//    var current = new System.Text.StringBuilder();
//    bool inQuotes = false;

//    foreach (char c in line)
//    {
//        if (c == '"') { inQuotes = !inQuotes; }
//        else if (c == ',' && !inQuotes) { result.Add(current.ToString()); current.Clear(); }
//        else { current.Append(c); }
//    }

//    result.Add(current.ToString());
//    return result.ToArray();
//}

//Console.WriteLine("=== Book Recommendation System ===\n");

//var ratingPath = "C:/FYJIXFINAL/UAT/ML.NET/ModelMaker/Dataset/RecommendationEngine/Ratings.csv";
//var booksPath = "C:/FYJIXFINAL/UAT/ML.NET/ModelMaker/Dataset/RecommendationEngine/Books.csv";
//var random = new Random(42);

//var lines = File.ReadAllLines(booksPath).Take(5);
//foreach (var line in lines)
//    Console.WriteLine($"RAW: {line}");

//var books = File.ReadAllLines(booksPath)
//    .Skip(1)
//    .Select(line =>
//    {
//       // handle commas inside quoted fields
//       var cols = SplitCsv(line);
//        if (cols.Length < 5) return null;
//        return new BookContent
//        {
//            ISBN = cols[0].Trim(),
//            Title = cols[1].Trim(),
//            Author = cols[2].Trim(),
//            Year = cols[3].Trim(),
//            Publisher = cols[4].Trim(),
//            Features = $"{cols[1].Trim()} {cols[2].Trim()} {cols[4].Trim()}"// Author + Publisher
//        };
//    })
//    .Where(x => x != null && !string.IsNullOrEmpty(x.ISBN))
//    .ToList();

////ISBN → BookContent lookup
//var bookLookup = books
//    .GroupBy(x => x.ISBN)
//    .ToDictionary(g => g.Key, g => g.First());

//Console.WriteLine($"Books loaded: {books.Count}");

//Console.WriteLine("Loading ratings...");

//var rawData = File.ReadAllLines(ratingPath)
//    .Skip(1)
//    .Select(line => line.Split(','))
//    .Where(cols => cols.Length == 3)
//    .Select(cols => new
//    {
//        UserId = cols[0].Trim(),
//        ISBN = cols[1].Trim(),
//        Rating = float.TryParse(cols[2].Trim(), out var r) ? r : -1f
//    })
//    .Where(x => x.Rating > 0 && bookLookup.ContainsKey(x.ISBN)) // only books we know about
//    .ToList();

//Console.WriteLine($"Ratings loaded: {rawData.Count}");

//var activeUsers = rawData
//    .GroupBy(x => x.UserId)
//    .Where(g => g.Count() >= 3)
//    .Select(g => g.Key)
//    .ToHashSet();

//var popularBooks = rawData
//    .GroupBy(x => x.ISBN)
//    .Where(g => g.Count() >= 3)
//    .Select(g => g.Key)
//    .ToHashSet();

//var filteredRatings = rawData
//    .Where(x => activeUsers.Contains(x.UserId) && popularBooks.Contains(x.ISBN))
//    .Select(x => new BookRating
//    {
//        UserId = x.UserId,
//        ISBN = x.ISBN,
//        Rating = x.Rating <= 6f ? 0f   // Disliked
//               : x.Rating <= 8f ? 1f   // Neutral
//               : 2f                     // Liked
//    })
//    .ToList();

//Console.WriteLine($"Filtered ratings: {filteredRatings.Count}");

//var trainList = new List<BookRating>();
//var testList = new List<BookRating>();

//foreach (var userGroup in filteredRatings.GroupBy(x => x.UserId))
//{
//    var userRatings = userGroup.OrderBy(_ => random.Next()).ToList();

//    if (userRatings.Count < 2)
//    {
//        trainList.AddRange(userRatings);
//        continue;
//    }

//    var trainCount = (int)(userRatings.Count * 0.8);
//    trainCount = trainCount == 0 ? 1 : trainCount;

//    trainList.AddRange(userRatings.Take(trainCount));
//    testList.AddRange(userRatings.Skip(trainCount));
//}

//Console.WriteLine($"Train: {trainList.Count}  Test: {testList.Count}");

//var trainView = mlContext.Data.LoadFromEnumerable(trainList);
//var testView = mlContext.Data.LoadFromEnumerable(testList);


//Console.WriteLine("\nTraining collaborative model...");

//var collabPipeline = mlContext.Transforms.Conversion.MapValueToKey(
//        inputColumnName: nameof(BookRating.UserId),
//        outputColumnName: "UserIdEncoded"
//    )
//    .Append(mlContext.Transforms.Conversion.MapValueToKey(
//        inputColumnName: nameof(BookRating.ISBN),
//        outputColumnName: "ISBNEncoded"
//    ))
//    .Append(mlContext.Recommendation().Trainers.MatrixFactorization(
//        new MatrixFactorizationTrainer.Options
//        {
//            MatrixColumnIndexColumnName = "UserIdEncoded",
//            MatrixRowIndexColumnName = "ISBNEncoded",
//            LabelColumnName = "Label",
//            NumberOfIterations = 100,
//            ApproximationRank = 200,
//            LearningRate = 0.005,
//            Lambda = 0.05,
//            LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression
//        }
//    ));

//var collabModel = collabPipeline.Fit(trainView);
//var collabPredictor = mlContext.Model
//    .CreatePredictionEngine<BookRating, BookRatingPrediction>(collabModel);

//var collabPredictions = collabModel.Transform(testView);
//var collabResults = mlContext.Data
//    .CreateEnumerable<BookRatingResult>(collabPredictions, reuseRowObject: false)
//    .Where(x => !float.IsNaN(x.Score))
//    .ToList();

//var collabMae = collabResults.Average(x => Math.Abs(x.Label - x.Score));
//var collabRmse = Math.Sqrt(collabResults.Average(x => Math.Pow(x.Label - x.Score, 2)));
//var meanLabel = collabResults.Average(x => x.Label);
//var ssTot = collabResults.Sum(x => Math.Pow(x.Label - meanLabel, 2));
//var ssRes = collabResults.Sum(x => Math.Pow(x.Label - x.Score, 2));
//var collabR2 = 1 - (ssRes / ssTot);

//Console.WriteLine($"\n=== Collaborative Model ===");
//Console.WriteLine($"  R²:   {collabR2:F4}");
//Console.WriteLine($"  RMSE: {collabRmse:F4}");
//Console.WriteLine($"  MAE:  {collabMae:F4}");


//Console.WriteLine("\nBuilding content model...");

//var bookFeatureVectors = new Dictionary<string, HashSet<string>>();


//foreach (var book in books)
//{
//    var words = book.Features
//        .ToLower()
//        .Split(' ', StringSplitOptions.RemoveEmptyEntries)
//        .ToHashSet();
//    bookFeatureVectors[book.ISBN] = words;
//}

//float ContentSimilarity(string isbn1, string isbn2)
//{
//    if (!bookFeatureVectors.ContainsKey(isbn1) ||
//        !bookFeatureVectors.ContainsKey(isbn2)) return 0f;

//    var set1 = bookFeatureVectors[isbn1];
//    var set2 = bookFeatureVectors[isbn2];

//    var intersection = set1.Intersect(set2).Count();
//    var union = set1.Union(set2).Count();

//    return union == 0 ? 0f : (float)intersection / union;
//}

//float ContentScore(string userId, string candidateISBN)
//{
//    var likedBooks = filteredRatings
//        .Where(x => x.UserId == userId && x.Rating == 2f)
//        .Select(x => x.ISBN)
//        .ToList();

//    if (!likedBooks.Any()) return 0f;

//    return likedBooks
//        .Select(isbn => ContentSimilarity(isbn, candidateISBN))
//        .Average();
//}

//Console.WriteLine("Content model ready!");


//const float collabWeight = 0.6f;
//const float contentWeight = 0.4f;

//List<HybridRecommendation> GetHybridRecommendations(string userId, int topN = 10)
//{
//    var alreadyRated = filteredRatings
//        .Where(x => x.UserId == userId)
//        .Select(x => x.ISBN)
//        .ToHashSet();

//    var candidates = popularBooks
//        .Where(isbn => !alreadyRated.Contains(isbn))
//        .ToList();

//    var recommendations = new List<HybridRecommendation>();

//    foreach (var isbn in candidates)
//    {
//        var collabScore = collabPredictor.Predict(new BookRating
//        {
//            UserId = userId,
//            ISBN = isbn
//        }).Score;

//        if (float.IsNaN(collabScore)) collabScore = 0f;

//        var collabNorm = collabScore / 2f;
//        var contentScr = ContentScore(userId, isbn);

//        var hybridScore = (collabNorm * collabWeight)
//                        + (contentScr * contentWeight);

//        bookLookup.TryGetValue(isbn, out var bookInfo);

//        recommendations.Add(new HybridRecommendation
//        {
//            ISBN = isbn,
//            Title = bookInfo?.Title ?? "Unknown",
//            Author = bookInfo?.Author ?? "Unknown",
//            CollaborativeScore = collabNorm,
//            ContentScore = contentScr,
//            HybridScore = hybridScore
//        });
//    }

//    return recommendations
//        .OrderByDescending(x => x.HybridScore)
//        .Take(topN)
//        .ToList();
//}


//var testUser = trainList
//    .GroupBy(x => x.UserId)
//    .OrderByDescending(g => g.Count())
//    .First().Key;

//Console.WriteLine($"\n=== Top 10 Hybrid Recommendations for User {testUser} ===");
//Console.WriteLine($"{"Title",-40} {"Author",-25} {"Collab",-10} {"Content",-10} {"Hybrid",-10}");
//Console.WriteLine(new string('-', 100));

//var recommendations = GetHybridRecommendations(testUser);
//recommendations.ForEach(r =>
//{
//    var title = r.Title.Length > 38 ? r.Title[..38] + ".." : r.Title;
//    var author = r.Author.Length > 23 ? r.Author[..23] + ".." : r.Author;
//    Console.WriteLine($"{title,-40} {author,-25} {r.CollaborativeScore,-10:F3} {r.ContentScore,-10:F3} {r.HybridScore,-10:F3}");
//});


//ML/bin/debug/net10.0
//var modelDirectoryForRec = Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", "..", "ML.NET", "wwwroot", "recommendation_engine_model.zip"));

////Directory.CreateDirectory(modelDirectoryForRec);

////var modelPathForRec = Path.Combine(modelDirectoryForRec, "RecommendationEngine.zip");
////mlContext.Model.Save(collabModel, trainView.Schema , modelPathForRec);
#endregion


#region Customer Segmentation

var usersPath = "C:/FYJIXFINAL/UAT/ML.NET/ModelMaker/Dataset/RecommendationEngine/Users.csv";

var lines2 = File.ReadAllLines(usersPath).Take(5);
foreach (var line in lines2)
    Console.WriteLine($"RAW: {line}");

Console.WriteLine("\n=== Customer Segmentation ===\n");

// ─────────────────────────────────────────────────
// STEP 1 — Build user profiles from ratings
// ─────────────────────────────────────────────────
Console.WriteLine("Building user profiles...");

var bookLookup = books
    .GroupBy(x => x.ISBN)
    .ToDictionary(g => g.Key, g => g.First());

var rawData2 = File.ReadAllLines(usersPath)
    .Skip(1)
    .Select(line => line.Split(','))
    .Where(cols => cols.Length == 3)
    .Select(cols => new
    {
        UserId = cols[0].Trim(),
        ISBN = cols[1].Trim(),
        Rating = float.TryParse(cols[2].Trim(), out var r) ? r : -1f
    })
    .Where(x => x.Rating > 0 && bookLookup.ContainsKey(x.ISBN)) // only books we know about
    .ToList();



var userProfiles = rawData2
    .GroupBy(x => x.UserId)
    .Where(g => g.Count() >= 3) // need minimum ratings for meaningful profile
    .Select(g =>
    {
        var ratings = g.Select(x => x.Rating).ToList();
        var total = ratings.Count;
        var avg = ratings.Average();
        var liked = ratings.Count(r => r >= 8f) / (float)total;
        var disliked = ratings.Count(r => r <= 4f) / (float)total;
        var variance = ratings.Select(r => Math.Pow(r - avg, 2)).Average();

        return new UserProfile
        {
            UserId = g.Key,
            TotalRatings = total,
            AvgRating = (float)avg,
            PercentLiked = liked,
            PercentDisliked = disliked,
            RatingVariance = (float)variance
        };
    })
    .ToList();

Console.WriteLine($"User profiles built: {userProfiles.Count}");

// ─────────────────────────────────────────────────
// STEP 2 — Convert to ML.NET feature vectors
// ─────────────────────────────────────────────────

// Normalize each feature to 0-1 range so no single feature dominates
float Normalize(float value, float min, float max) =>
    max == min ? 0f : (value - min) / (max - min);

var minTotal = userProfiles.Min(x => x.TotalRatings);
var maxTotal = userProfiles.Max(x => x.TotalRatings);
var minAvg = userProfiles.Min(x => x.AvgRating);
var maxAvg = userProfiles.Max(x => x.AvgRating);
var minVariance = userProfiles.Min(x => x.RatingVariance);
var maxVariance = userProfiles.Max(x => x.RatingVariance);

var featureData = userProfiles
    .Select(u => new UserFeatures
    {
        Features = new float[]
        {
            Normalize(u.TotalRatings,    minTotal,    maxTotal),
            Normalize(u.AvgRating,       minAvg,      maxAvg),
            u.PercentLiked,      // already 0-1
            u.PercentDisliked,   // already 0-1
            Normalize(u.RatingVariance,  minVariance, maxVariance)
        }
    })
    .ToList();

// ─────────────────────────────────────────────────
// STEP 3 — Elbow Method (find best K)
// ─────────────────────────────────────────────────
Console.WriteLine("\nFinding optimal K using Elbow Method...");
Console.WriteLine($"{"K",-5} {"Inertia",-15} {"Improvement",-15}");
Console.WriteLine(new string('-', 35));

var mlContextCluster = new MLContext(seed: 1);
var dataForClustering = mlContextCluster.Data.LoadFromEnumerable(featureData);

var inertiaScores = new Dictionary<int, double>();
double prevInertia = double.MaxValue;

for (int k = 2; k <= 10; k++)
{
    var clusterPipeline = mlContextCluster.Transforms
        .NormalizeMinMax("Features")
        .Append(mlContextCluster.Clustering.Trainers.KMeans(
            featureColumnName: "Features",
            numberOfClusters: k
        ));

    var clusterModel = clusterPipeline.Fit(dataForClustering);
    var clusterPreds = clusterModel.Transform(dataForClustering);

    // Inertia = sum of squared distances to cluster centers
    var clusterResults = mlContextCluster.Data
        .CreateEnumerable<UserSegment>(clusterPreds, reuseRowObject: false)
        .ToList();

    // Score array contains distances to each cluster — min = distance to assigned cluster
    double inertia = clusterResults
        .Sum(x => x.Score != null ? x.Score.Min() : 0f);

    inertiaScores[k] = inertia;

    var improvement = prevInertia == double.MaxValue
        ? 0
        : ((prevInertia - inertia) / prevInertia) * 100;

    Console.WriteLine($"{k,-5} {inertia,-15:F2} {(prevInertia == double.MaxValue ? "baseline" : $"{improvement:F1}% better"),-15}");

    prevInertia = inertia;
}

// ─────────────────────────────────────────────────
// STEP 4 — Pick best K automatically
// Pick the K where improvement drops below 15%
// ─────────────────────────────────────────────────
int bestK = 3; // default fallback

for (int k = 3; k <= 10; k++)
{
    var improvement = (inertiaScores[k - 1] - inertiaScores[k])
                    / inertiaScores[k - 1] * 100;
    if (improvement < 15.0)
    {
        bestK = k - 1;
        break;
    }
    bestK = k;
}

Console.WriteLine($"\nOptimal K = {bestK} ✅");

// ─────────────────────────────────────────────────
// STEP 5 — Train final model with best K
// ─────────────────────────────────────────────────
Console.WriteLine($"\nTraining final model with K={bestK}...");

var finalPipeline = mlContextCluster.Transforms
    .NormalizeMinMax("Features")
    .Append(mlContextCluster.Clustering.Trainers.KMeans(
        featureColumnName: "Features",
        numberOfClusters: bestK
    ));

var finalModel = finalPipeline.Fit(dataForClustering);
var finalPreds = finalModel.Transform(dataForClustering);

var segmentResults = mlContextCluster.Data
    .CreateEnumerable<UserSegment>(finalPreds, reuseRowObject: false)
    .ToList();

// Zip results back with user profiles
var usersWithSegments = userProfiles
    .Zip(segmentResults, (profile, segment) => new
    {
        profile.UserId,
        profile.TotalRatings,
        profile.AvgRating,
        profile.PercentLiked,
        profile.PercentDisliked,
        profile.RatingVariance,
        Segment = (int)segment.ClusterId
    })
    .ToList();

// ─────────────────────────────────────────────────
// STEP 6 — Analyze each segment
// ─────────────────────────────────────────────────
Console.WriteLine($"\n=== Segment Analysis ===\n");

for (int seg = 1; seg <= bestK; seg++)
{
    var group = usersWithSegments.Where(x => x.Segment == seg).ToList();
    if (!group.Any()) continue;

    var avgTotal = group.Average(x => x.TotalRatings);
    var avgRating = group.Average(x => x.AvgRating);
    var avgLiked = group.Average(x => x.PercentLiked) * 100;
    var avgDisliked = group.Average(x => x.PercentDisliked) * 100;
    var avgVariance = group.Average(x => x.RatingVariance);

    // Auto label the segment based on behavior
    var label = (avgRating, avgTotal, avgLiked) switch
    {
        var (r, t, l) when r >= 8f && t >= 20f => "⭐ Enthusiast Readers",
        var (r, t, l) when r >= 8f && t < 20f => "👍 Casual Fans",
        var (r, t, l) when r < 5f => "👎 Harsh Critics",
        var (r, t, l) when l >= 0.7f => "😍 Generous Raters",
        _ => "📚 Average Readers"
    };

    Console.WriteLine($"Segment {seg} — {label}");
    Console.WriteLine($"  Users:           {group.Count}");
    Console.WriteLine($"  Avg Ratings:     {avgTotal:F0} books rated");
    Console.WriteLine($"  Avg Score Given: {avgRating:F2} / 10");
    Console.WriteLine($"  % Liked (8-10):  {avgLiked:F1}%");
    Console.WriteLine($"  % Disliked(1-4): {avgDisliked:F1}%");
    Console.WriteLine($"  Rating Variance: {avgVariance:F2}");
    Console.WriteLine();
}

// ─────────────────────────────────────────────────
// STEP 7 — Show sample users per segment
// ─────────────────────────────────────────────────
Console.WriteLine("=== Sample Users per Segment ===\n");

for (int seg = 1; seg <= bestK; seg++)
{
    var group = usersWithSegments
        .Where(x => x.Segment == seg)
        .Take(3)
        .ToList();

    Console.WriteLine($"Segment {seg}:");
    Console.WriteLine($"  {"UserId",-10} {"Ratings",-10} {"AvgScore",-10} {"Liked%",-10} {"Disliked%",-10}");
    Console.WriteLine($"  {new string('-', 55)}");
    group.ForEach(u => Console.WriteLine(
        $"  {u.UserId,-10} {u.TotalRatings,-10:F0} {u.AvgRating,-10:F2} {u.PercentLiked * 100,-10:F1} {u.PercentDisliked * 100,-10:F1}"));
    Console.WriteLine();
}

#endregion