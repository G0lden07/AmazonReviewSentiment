using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace AmazonReviewSentiment
{
    public class SentimentData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool Sentiment;

        [LoadColumn(1)]
        public string? Title;

        [LoadColumn(2)]
        public string? ReviewText;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
