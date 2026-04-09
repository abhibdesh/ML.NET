using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace BLModels
{
     public class SentimentAnalysisData
    {
        [LoadColumn(0)] 
        public string Year { get; set; }
        [LoadColumn(1)] 
        public string Month { get; set; }
        [LoadColumn(2)] 
        public string Day { get; set; }
        [LoadColumn(3)] 
        public string TimeofTweet { get; set; }
        [LoadColumn(4)] 
        public string text { get; set; }

        [LoadColumn(5), ColumnName("Label")] 
        public string sentiment { get; set; }
        [LoadColumn(6)] 
        public string Platform { get; set; }
    }

    public class SentimentAnalysisPrediction
    {
        public string PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }
}
