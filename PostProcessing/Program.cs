//
// Program.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2016 University of Bristol
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace PostProcessing
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using BayesianDictionaryLearning;
    using System.Linq;
    using MicrosoftResearch.Infer.Maths;
    using BayesianDictionaryLearning.Models;
    using InferHelpers;
    using MicrosoftResearch.Infer.Collections;
    using MicrosoftResearch.Infer.Learners;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Learners.Mappings;

    internal class Program
    {
        private static void Main(string[] args)
        {
            // ConvergenceHelper();
            // NoisyData();
            // Sparsity();
            // Classification("accel");
            Classification("sphere");
            // Reconstruct();
        }

        private static void NoisyData()
        {
            string current = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory("..");

            const int nTrain = int.MaxValue;
            const int nTest = 1;
            const int bases = 128;
            const bool sparse = false;
            const bool normConstraints = false;
            const bool bias = false;

            var data = Data.LoadTrainTestData("accel", nTrain, nTest, false, false);

            string filename = $"../../marginals/marginals_accel_full_sparse={sparse}_nc={normConstraints}_bias={bias}_{bases}_bases.json";
            var marginalsCollection = MarginalsCollection.Load(filename);

            var priors = new Marginals
            {
                Dictionary = marginalsCollection.TrainPosteriors.Dictionary
            };

            Rand.Restart(0);

            var models = new ModelCollection
            {
                TrainFixed =
                    new BDLSimple(new BDLParameters
                    {
                        Mode = Mode.TrainFixed,
                        Sparse = sparse,
                        NormConstraints = normConstraints,
                        UseMatrixMultiply = false,
                        MissingData = true,
                        Debug = true
                    }),
                Reconstruct =
                    new BDLSimple(new BDLParameters
                    {
                        Mode = Mode.Reconstruct,
                        Sparse = sparse,
                        NormConstraints = normConstraints,
                        UseMatrixMultiply = false,
                        MissingData = true
                    }),
            };

            models.TrainFixed.ConstructModel();

            // We'll randomly remove points from the test data, and see if we can recover it
            var missingRates = new[] {0.1};

            var results = new Results {Name = "missing"};

            foreach (double missingRate in missingRates)
            {
                var testData =
                    data.TestSignals.EnumerateRows()
                        .Select(row => row.Select(ia => Rand.Double() > missingRate ? ia : double.NaN).ToArray()).ToArray();

                // Plot noisy signals
                // PlottingHelper.PlotFunctions(testData, "Missing data", string.Empty, 1, 1, 1);

                var testPosteriors = models.TrainFixed.Train(priors, testData);
                var signals = models.Reconstruct.Reconstruct(testPosteriors).Signals;
                marginalsCollection.Reconstructed = models.Reconstruct.Reconstruct(
                                        marginalsCollection.TestPosteriors).Signals;

                var reconstructions = data.TestSignals.ToRowArrays().Zip(marginalsCollection.Reconstructed,
                                        (s, e) => new Reconstruction {Signal = s, Estimate = e}).ToArray();

                // Compute average reconstruction error
                results.Errors.Add(reconstructions.Select(ia => ia.Error).ToArray().Average());

                string message = $"Reconstructions (missing rate {missingRate}), avg. error={results.Errors.Last():N4}";
                Console.WriteLine(message);

                PlottingHelper.PlotReconstructions(reconstructions, results.Errors.Last(), 1, 1, 1, string.Empty, false);
            }

            results.Save(MainClass.ResultPath);
            Directory.SetCurrentDirectory(current);
        }

        private static void ConvergenceHelper()
        {
            string current = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory("..");

            var errors = new Dictionary<string, IEnumerable<double>>();
            var evidence = new Dictionary<string, IEnumerable<double>>();

            foreach (bool norm in new[] { false })
            {
                foreach (bool sparse in new[] { false, true })
                {
                    foreach (bool nc in new[] { false })
                    {
                        foreach (bool bias in new[] { false })
                        {
                            string filename = $"../../results/BayesianDictionaryLearning.Results_Convergence nrm={norm}, sparse={sparse}, nc={nc}, bias={bias}.json";
                            var convergenceResults = Results.Load(filename);

                            string s = sparse ? "sparse" : "non-sparse";
                            string n = nc ? " n.c." : string.Empty;
                            errors[$"{s}{n}"] = convergenceResults.Errors;
                            evidence[$"{s}{n}"] = convergenceResults.Evidence;

//                            Plotter.TwinPlot(
//                                ArrayHelpers.DoubleRange(0, convergenceResults.Errors.Count),
//                                convergenceResults.Errors,
//                                convergenceResults.Evidence,
//                                string.Empty,
//                                "Iterations",
//                                "RMSE",
//                                "Evidence log odds");
                        }
                    }
                }
            }

//            PlottingHelper.TwinTwinPlot(
//                errors,
//                evidence,
//                "BDL Convergence",
//                "Iterations",
//                "RMSE",
//                "Evidence log odds",
//                true);

            // PlottingHelper.Plot(errors, "Convergence of reconstruction error", string.Empty, "Iterations", "RMSE", true);
            // PlottingHelper.Plot(evidence, "Convergence of model evidence", string.Empty, "Iterations",
            //    "Evidence log odds", true);
            PlottingHelper.Plot(errors, "Convergence of reconstruction error", "Iterations", "RMSE", true);
            PlottingHelper.Plot(evidence, "Convergence of model evidence", "Iterations", "Evidence log odds", true);
            Directory.SetCurrentDirectory(current);
        }

        private static void Sparsity()
        {
            string current = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory("..");
            const int nTrain = int.MaxValue;
            const int nTest = 1;
            const bool normConstraints = false;
            const bool useBias = false;
            var data = Data.LoadTrainTestData("accel", nTrain, nTest, false, false);

            var indices = new int[6];
            for (var i = 0; i < 6; i++)
            {
                indices[i] = data.TrainLabels.IndexOf((double)i);
            }

            var classes = new[]
            {
                "Walking",
                "Ascending stairs",
                "Descending stairs",
                "Sitting",
                "Standing",
                "Lying down"
            };

            // Console.WriteLine($"{string.Join(", ", indices)}");
            // foreach (int numBases in new[] {64, 128, 256, 512})
            foreach (int numBases in new[] {128})
            {
                foreach (bool sparse in new[] {true, false})
                {
                    string st = $"{data.Name} full_sparse={sparse}_nc={normConstraints}_bias={useBias} {numBases} bases";

                    // string filename =
                    //    $"../../marginals/marginals_accel_full_sparse={sparse}_nc={normConstraints}_bias={useBias}_{numBases}_bases.json";
                    string filename =
                        $"../../marginals/BayesianDictionaryLearning.MarginalsCollection_accel full_sparse={sparse}_nc=False_bias=False {numBases} bases.json";
                    var marginalsCollection = MarginalsCollection.Load(filename);
                    var coefficients = marginalsCollection.TrainPosteriors.Coefficients.Take(50).ToArray();
                    double sparsity = coefficients.GetSparsity(1e-4).Average();
                    string title = $"Coefficients, average. sparsity {sparsity:N2}";
                    Console.WriteLine(title);

                    // continue;
                    PlottingHelper.SparsityPlot(coefficients, title, $"sparsity_sparse={sparse}", true);

                    continue;
                    // TODO: Put the class names as titles for each of the subplots
                    PlottingHelper.PlotResults(numBases, data.TrainSignals.ColumnCount,
                        marginalsCollection.TrainPosteriors.Dictionary,
                        indices.Select(i => marginalsCollection.TrainPosteriors.Coefficients[i]).ToArray(),
                        st, false, true, true);
                }
            }

            Directory.SetCurrentDirectory(current);
        }

        private static void Classification(string dataset)
        {
            string current = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory("..");
            const int nTrain = int.MaxValue;
            const int nTest = int.MaxValue;
            const bool normConstraints = false;
            const bool useBias = false;
            double lambda = (dataset == "accel") ? 0.1 : 0.03;
            const bool spams = true;

            var data = Data.LoadTrainTestData(dataset, nTrain, nTest, false, false);

            var activities = dataset == "accel"
                ? new[]
                {
                    "Walking",
                    "Ascending stairs",
                    "Descending stairs",
                    "Sitting",
                    "Standing",
                    "Lying down"
                }
                : new[] {"Lying down", "Standing", "Walking"};

            Func<double[], Vector> toVectorWithBias = ia => Vector.FromArray(ia.Concat(new[] {1.0}).ToArray());
            var trainLabels = data.TrainLabels.Select(ia => activities[(int) ia]).ToArray();
            var testLabels = data.TestLabels.Select(ia => activities[(int) ia]).ToArray();

            foreach (int numBases in new[] { 64 })
            {
                if (spams)
                {
                    var precomputed = Precomputed.Load(dataset, numBases, lambda, false, false, true, nTrain, nTest, false);
                    var trainSpams =
                        precomputed.TrainCoefficients.EnumerateRows()
                            .Take(7802)
                            .Select(ia => toVectorWithBias(ia.ToArray()))
                            .ToArray();
                    var testSpams =
                        precomputed.TestCoefficients.EnumerateRows()
                            .Select(ia => toVectorWithBias(ia.ToArray()))
                            .ToArray();

                    Console.WriteLine($"Train X: {trainSpams.Length}, y: {trainLabels.Length}");
                    Console.WriteLine($"Test  X: {testSpams.Length}, y: {testLabels.Length}");

                    BayesPointMachine(trainSpams, trainLabels, testSpams, testLabels, activities, $"spams {numBases}");
                }

                foreach (bool sparse in new[] {false, true})
                {
                    string st = $"{data.Name} full_sparse={sparse}_nc={normConstraints}_bias={useBias} {numBases} bases";
                    // BayesianDictionaryLearning.MarginalsCollection_accel full_sparse=False_nc=False_bias=False 64 bases

                    // string filename =
                    //     $"../../marginals/marginals_accel_full_sparse={sparse}_nc={normConstraints}_bias={useBias}_{numBases}_bases.json";
                    // BayesianDictionaryLearning.MarginalsCollection_accel full_sparse=True_nc=False_bias=False 64 bases
                    string filename = dataset == "accel"
                        ? $"../../marginals/BayesianDictionaryLearning.MarginalsCollection_{dataset} full_sparse={sparse}_nc={normConstraints}_bias={useBias} {numBases} bases.json"
                        : $"../../marginals/BayesianDictionaryLearning.MarginalsCollection_{dataset} sphere_sparse={sparse} {numBases} bases.json";
                    var marginalsCollection = MarginalsCollection.Load(filename);

                    var trainFeatures =
                        marginalsCollection.TrainPosteriors.Coefficients.GetMeans<Gaussian>()
                            .Select(toVectorWithBias)
                            .ToArray();

                    var testFeatures =
                        marginalsCollection.TestPosteriors.Coefficients.GetMeans<Gaussian>()
                            .Select(toVectorWithBias)
                            .ToArray();


                    BayesPointMachine(trainFeatures, trainLabels, testFeatures, testLabels, activities,
                        $"sparse={sparse}_nc={normConstraints}_bias={useBias} {numBases} bases");
                }
            }

            Directory.SetCurrentDirectory(current);
        }

        private static void BayesPointMachine(IList<Vector> trainFeatures, IList<string> trainLabels,
            IList<Vector> testFeatures, IList<string> testLabels, IEnumerable<string> activities, string subTitle)
        {
            var mapping = new ClassifierMapping();
            var classifier = BayesPointMachineClassifier.CreateMulticlassClassifier(mapping);

            // Train the Bayes Point Machine classifier on the accelerometer data, including a bias feature
            Console.WriteLine("Training");
            classifier.Train(trainFeatures, trainLabels);
            Console.WriteLine("Done");

            // Create an evaluator for mapping
            var evaluatorMapping = mapping.ForEvaluation();
            var evaluator = new ClassifierEvaluator<IList<Vector>, int, IList<string>, string>(evaluatorMapping);

            // Evaluate on test set
            Console.WriteLine("Predicting");
            var predictions = classifier.PredictDistribution(testFeatures).ToArray();
            var estimates = classifier.Predict(testFeatures);
            Console.WriteLine("Done");

            double errorCount = evaluator.Evaluate(testFeatures, testLabels, estimates, Metrics.ZeroOneError);
            Console.WriteLine($"Error count: {errorCount}/{testLabels.Count}");

            foreach (string activity in activities)
            {
                double areaUnderRocCurve = evaluator.AreaUnderRocCurve(activity, testFeatures, testLabels,
                    predictions);
                var rocCurve =
                    evaluator.ReceiverOperatingCharacteristicCurve(activity, testFeatures, predictions).ToArray();
                PlottingHelper.Plot(
                    rocCurve.Select(ia => ia.First),
                    rocCurve.Select(ia => ia.Second),
                    $"ROC: {activity}, AUC={areaUnderRocCurve:N2}",
                    subTitle,
                    "False positive rate",
                    "True positive rate",
                    false);
                Console.WriteLine($"{activity}, AUC={areaUnderRocCurve:N2}");
            }
        }

        private static void Reconstruct()
        {
            string current = Environment.CurrentDirectory;
            Directory.SetCurrentDirectory("..");
            const bool normConstraints = false;
            const bool useBias = false;
            const int numBases = 128;
            const bool sparse = false;

            var data = Data.LoadTrainTestData("accel", 1, 2, false, false);
            string st = $"{data.Name} full {numBases} bases";

            string filename =
                $"../../marginals/BayesianDictionaryLearning.MarginalsCollection_accel full_sparse={sparse}_nc={normConstraints}_bias={useBias} {numBases} bases.json";
            var marginalsCollection = MarginalsCollection.Load(filename);

            var reconstructions = data.TestSignals.ToRowArrays().Zip(marginalsCollection.Reconstructed,
                (s, e) => new Reconstruction {Signal = s, Estimate = e}).ToArray();

            // Compute average reconstruction error
            double error = reconstructions.Select(ia => ia.Error).ToArray().Average();
            Console.WriteLine($"Reconstruction error: {error}");

            // plot normalised reconstructions
            PlottingHelper.PlotReconstructions(reconstructions, error, 2, 2, 1, st, false, true);

            Directory.SetCurrentDirectory(current);
        }
    }
}
