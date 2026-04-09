using Microsoft.ML.Data;

namespace BLModels
{
    // ── Collaborative (Ratings) ──────────────────
    public class BookRating
    {
        [LoadColumn(0)] public string UserId { get; set; }
        [LoadColumn(1)] public string ISBN { get; set; }

        [LoadColumn(2), ColumnName("Label")]
        public float Rating { get; set; }
    }

    public class BookRatingPrediction
    {
        public float Score { get; set; }
    }

    public class BookRatingResult
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }

    // ── Content (Books) ──────────────────────────
    public class BookContent
    {
        public string ISBN { get; set; }
        public string Title { get; set; }
        public string Author { get; set; }
        public string Year { get; set; }
        public string Publisher { get; set; }
        public string Features { get; set; } // Author + Publisher combined
    }

    // ── Hybrid Result ────────────────────────────
    public class HybridRecommendation
    {
        public string ISBN { get; set; }
        public string Title { get; set; }
        public string Author { get; set; }
        public float CollaborativeScore { get; set; }
        public float ContentScore { get; set; }
        public float HybridScore { get; set; }
    }
}