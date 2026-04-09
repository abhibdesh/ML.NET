using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace BLModels
{
    public class UserFeatures
    {
        [VectorType(5)]
        public float[] Features { get; set; }
    }

    public class UserSegment
    {
        public float[] Features { get; set; }
        public uint ClusterId { get; set; }
        public float[] Score { get; set; } // distance to each cluster center
    }

    public class UserProfile
    {
        public string UserId { get; set; }
        public float TotalRatings { get; set; }
        public float AvgRating { get; set; }
        public float PercentLiked { get; set; }
        public float PercentDisliked { get; set; }
        public float RatingVariance { get; set; }
    }
}
