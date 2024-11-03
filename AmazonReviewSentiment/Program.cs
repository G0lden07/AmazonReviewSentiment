using Microsoft.ML;
using Microsoft.ML.Data;
using AmazonReviewSentiment;
using static Microsoft.ML.DataOperationsCatalog;
using ExcelDataReader;
string _trainingDataPath = Path.Combine("C:\\Users\\Personal Files\\Documents\\GitHub\\AmazonReviewSentiment\\AmazonReviewSentiment\\", "Data", "train-reviews-micro.xlsx");
string _testingDataPath = Path.Combine("C:\\Users\\Personal Files\\Documents\\GitHub\\AmazonReviewSentiment\\AmazonReviewSentiment\\", "Data", "test-reviews-micro.xlsx");

MLContext mlContext = new MLContext();

var trainingDataList = LoadSentimentData(_trainingDataPath);
IDataView trainSet = mlContext.Data.LoadFromEnumerable(trainingDataList);

var testingDataList = LoadSentimentData(_testingDataPath);
IDataView testSet = mlContext.Data.LoadFromEnumerable(testingDataList);

ITransformer model = BuildAndTrainModel(mlContext, trainSet);
Evaluate(mlContext, model, testSet);

static List<SentimentData> LoadSentimentData(string filePath)
{
    System.Text.Encoding.RegisterProvider(System.Text.CodePagesEncodingProvider.Instance);
    using var stream = File.Open(filePath, FileMode.Open, FileAccess.Read);
    using var reader = ExcelReaderFactory.CreateReader(stream);
    var result = reader.AsDataSet();

    var sentimentDataList = new List<SentimentData>();
    var table = result.Tables[0]; // Assuming the first table contains your data

    for (int row = 1; row < table.Rows.Count; row++) // Assuming the first row is the header
    {
        var currentRow = table.Rows[row];
        var sentimentValue = Convert.ToInt32(currentRow[0]); // Read the sentiment value

        sentimentDataList.Add(new SentimentData
        {
            Sentiment = sentimentValue == 2, // Convert 2 to true and 1 to false
            Title = currentRow[1]?.ToString(), // Assuming the second column is Title
            ReviewText = currentRow[2]?.ToString() // Assuming the third column is ReviewText
        });
    }

    return sentimentDataList;
}

ITransformer BuildAndTrainModel(MLContext mlContext, IDataView splitTrainSet)
{
    var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.ReviewText))
        .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    Console.WriteLine("=============== Create and Train the Model ===============");
    var model = estimator.Fit(splitTrainSet);
    Console.WriteLine("=============== End of training ===============");
    Console.WriteLine();

    return model;
}

void Evaluate(MLContext mlContext, ITransformer model, IDataView splitTestSet)
{
    Console.WriteLine("=============== Evaluating Model accuracy with Test data===============");
    IDataView predictions = model.Transform(splitTestSet);
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

    Console.WriteLine();
    Console.WriteLine("Model quality metrics evaluation");
    Console.WriteLine("--------------------------------");
    Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    Console.WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    Console.WriteLine($"F1Score: {metrics.F1Score:P2}");
    Console.WriteLine("=============== End of model evaluation ===============");
}