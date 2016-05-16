//
// Runner.cs
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


namespace BayesianDictionaryLearning
{
    using System;
    using System.IO;
    using System.Linq;
    using System.Collections.Generic;
    using Models;
    using InferHelpers;
    using MicrosoftResearch.Infer.Distributions;
    using PythonPlotter;

    public static class Runner
    {
        public static Experiment RunningExperiment { get; set; } = new Experiment();

        public static void Toy()
        {
            var models = ModelCollection.CreateModels();
            const int n = 100;
            const double noise = 0.1;
            var toy = Data.GetToyData(128, n, 0.1);
            var bases = new[] {32, 64, 128, 256, 512, 1024, 2048};
            var resultsToy = new ResultsCollection
            {
                Name = toy.Name,
                NumTrain = n*2,
                NumTest = n*2,
                Bases = bases
            };
            resultsToy.Results.Add(Run(toy, ia => null, models, bases, $"{noise}", false));
            PlottingHelper.PlotErrorsWithEvidence(resultsToy.Results[0]);
            resultsToy.Save(MainClass.ResultPath);
        }

        public static void AcceleromterSphere()
        {
            var bases = new[] { 64 };
            const bool runSpams = false;
            const int n = int.MaxValue;
            const double lambda = 0.03;
            const bool normalise = false;

            var data = Data.LoadTrainTestData("sphere", n, n, false, false);
            data.TrainSignals = data.TrainSignals.NormalizeRows(2.0);
            data.TestSignals = data.TestSignals.NormalizeRows(2.0);

            var resultsSphere = new ResultsCollection
            {
                Name = "sphere",
                Normalised = normalise,
                NumTrain = data.TrainSignals.RowCount,
                NumTest = data.TestSignals.RowCount,
                Bases = bases
            };

            Func<int, Precomputed> loader = ia => null;
            if (runSpams)
            {
                loader = i => Precomputed.Load("accel", i, lambda, false, false, true, n, n, false);
            }

            foreach (bool sparse in new[] { false, true })
            {
                var models = ModelCollection.CreateModels(sparse, false, false);

                // Use SPAMS dictionary (fixed)
                if (runSpams)
                {
                    var trainModel = models.Train;
                    models.Train = models.TrainFixed;
                    resultsSphere.Results.Add(Run(data, loader, models, bases, "fixed", normalise));
                    models.Train = trainModel;
                }

                // Learn full model
                resultsSphere.Results.Add(Run(data, ia => null, models, bases, $"sphere_sparse={sparse}", normalise));
            }

            // Now plot the results of the different methods against each other
            resultsSphere.Save(MainClass.ResultPath);
        }

        public static void AccelerometerMain()
        {
            const bool runSpams = false;
            const bool runSeeded = false;
            const int n = int.MaxValue;
            // const int n = 10;
            var bases   = new[] { 64, 128, 256, 512 }; // , 1024 };
            // var bases = new[] {1024};
            // var bases   = new[] { 128, 256 };
            // var bases = new[] { 64, 128 };

            const double lambda = 0.1;

            foreach (bool normalise in new[] {false }) //, true})
            {
                // TODO: Test basic classification (best reconstruction?)
                // TODO: Try spike and slab if there's time?
                var data = Data.LoadTrainTestData("accel", n, n, false, false);
                if (normalise)
                {
                    data.TrainSignals = data.TrainSignals.NormalizeRows(2.0);
                    data.TestSignals = data.TestSignals.NormalizeRows(2.0);
                }

                var resultsAccel = new ResultsCollection
                {
                    Name = "accel",
                    Normalised = normalise,
                    NumTrain = data.TrainSignals.RowCount,
                    NumTest = data.TestSignals.RowCount,
                    Bases = bases
                };

                // TODO: Inneficient since we're loading everyting, and more than once!
                // TODO: SPAMS should have been learnt on normalised data to work here!
                Func<int, Precomputed> loader = ia => null;
                if (runSpams || runSeeded)
                {
                    loader = i => Precomputed.Load("accel", i, lambda, false, false, true, n, n, false);
                }

                // Pure spams results
                // resultsAccel.Results.Add(Run(data, loader, null, bases, "spams", normalise));

                foreach (bool sparse in new[] {true, false})
                {
                    foreach (bool normConstraints in new[] {false})
                    {
                        foreach (bool useBias in new[] {false})
                        {
                            var models = ModelCollection.CreateModels(sparse, normConstraints, useBias);

                            // Use SPAMS dictionary (fixed)
                            if (runSpams)
                            {
                                var trainModel = models.Train;
                                models.Train = models.TrainFixed;
                                resultsAccel.Results.Add(Run(data, loader, models, bases, "fixed", normalise));
                                models.Train = trainModel;
                            }

                            // Use SPAMS dictionary (seeded)
                            if (runSeeded)
                            {
                                resultsAccel.Results.Add(Run(data, loader, models, bases, "seeded", normalise));
                            }

                            // Learn full model
                            resultsAccel.Results.Add(Run(data, ia => null, models, bases,
                                $"full_sparse={sparse}_nc={normConstraints}_bias={useBias}",
                                normalise));
                        }
                    }

                }

                // Now plot the results of the different methods against each other
                resultsAccel.Save(MainClass.ResultPath);
            }
        }

        public static void EffectOfBases()
        {
            const int n = int.MaxValue;
            var bases = new[] {64, 128, 256, 512, 1024};

            // SPAMS regularisation parameter
            const double lambda = 0.1;

            foreach (bool normalise in new[] {false, true})
            {
                // TODO: Test basic classification (best reconstruction?)
                // TODO: Try spike and slab if there's time?
                var data = Data.LoadTrainTestData("accel", n, n, false, false);
                if (normalise)
                {
                    data.TrainSignals = data.TrainSignals.NormalizeRows(2.0);
                    data.TestSignals = data.TestSignals.NormalizeRows(2.0);
                }

                var resultsAccel = new ResultsCollection
                {
                    Name = "accel",
                    Normalised = normalise,
                    NumTrain = data.TrainSignals.RowCount,
                    NumTest = data.TestSignals.RowCount,
                    Bases = bases
                };

                // TODO: Inneficient since we're loading everyting, and more than once!
                // TODO: SPAMS should have been learnt on normalised data to work here!
                Func<int, Precomputed> loader =
                    i => Precomputed.Load("accel", i, lambda, false, false, true, n, n, false);

                // Pure spams results
                resultsAccel.Results.Add(Run(data, loader, null, bases, "spams", normalise));

                foreach (bool normConstraints in new[] {false, true})
                {
                    foreach (bool sparse in new[] {true, false})
                    {
                        foreach (bool useBias in new[] {false})
                        {
                            var models = ModelCollection.CreateModels(sparse, normConstraints, useBias);

                            // Learn full model
                            resultsAccel.Results.Add(Run(data, ia => null, models, bases,
                                $"full_sparse={sparse}_nc={normConstraints}_bias={useBias}",
                                normalise));
                        }
                    }
                }

                // Now plot the results of the different methods against each other

                resultsAccel.Save(MainClass.ResultPath);
            }
        }

        public static void Convergence()
        {
            // const int nTrain = int.MaxValue;
            const int nTrain = 1000;
            const int nTest = 200;
            var bases = new[] { 128 };

            var errors = new Dictionary<string, IEnumerable<double>>();
            var evidence = new Dictionary<string, IEnumerable<double>>();

            foreach (bool normalise in new[] {false }) //, true})
            {
                RunningExperiment = new Experiment
                {
                    Normalise = normalise,
                    DataSet = Data.LoadTrainTestData("accel", nTrain, nTest, false)
                };

                if (normalise)
                {
                    RunningExperiment.DataSet.TrainSignals = RunningExperiment.DataSet.TrainSignals.NormalizeRows(2.0);
                    RunningExperiment.DataSet.TestSignals = RunningExperiment.DataSet.TestSignals.NormalizeRows(2.0);
                }

                foreach (bool sparse in new[] {true })
                {
                    foreach (bool normConstraints in new[] { false }) //, false})
                    {
                        foreach (bool useBias in new[] {false}) //, true })
                        {
                            foreach (bool useMatrixMultiply in new[] {false, true})
                            {
                                string st =
                                    $"Convergence nrm={normalise}, sparse={sparse}, nc={normConstraints}, bias={useBias}, mm={useMatrixMultiply}";
                                Console.WriteLine(st);
                                RunningExperiment.Models = ModelCollection.CreateModels(sparse, normConstraints, useBias);
                                var handlers = new InferenceProgressHandlers {RunningExperiment = RunningExperiment};
                                ((BDLSimple) RunningExperiment.Models.Train).Parameters.ConvergenceCriterion =
                                    (monitor, d) => false;
                                ((BDLSimple) RunningExperiment.Models.Train).Parameters.MaxIterations[Mode.Train] = 20;

                                ((BDLSimple) RunningExperiment.Models.Train).Parameters.UseMatrixMultiply = useMatrixMultiply;
                                ((BDLSimple) RunningExperiment.Models.TrainFixed).Parameters.UseMatrixMultiply = useMatrixMultiply;
                                ((BDLSimple) RunningExperiment.Models.Reconstruct).Parameters.UseMatrixMultiply = useMatrixMultiply;

                                RunningExperiment.Models.Train.AddUpdateHandler(handlers.CustomHandler);
                                RunningExperiment.ConvergenceResults = new Results {Name = st};

                                var priors = Marginals.CreatePriors(null, null,
                                    RunningExperiment.DataSet.TrainSignals.ColumnCount, bases[0], 1.0, false);
                                RunningExperiment.Models.Train.Train(priors,
                                    RunningExperiment.DataSet.TrainSignals.ToRowArrays());
                                RunningExperiment.ConvergenceResults.Save(MainClass.ResultPath);

                                string s = sparse ? "sparse" : "non-sparse";
                                string n = normConstraints ? " n.c." : string.Empty;
                                string m = useMatrixMultiply ? "mm" : string.Empty;
                                errors[$"{s}{n}{m}"] = RunningExperiment.ConvergenceResults.Errors;
                                evidence[$"{s}{n}{m}"] = RunningExperiment.ConvergenceResults.Evidence;
                            }
                        }
                    }
                }
            }

            PlottingHelper.Plot(errors, "Convergence of test error", string.Empty, "Iterations", "RMSE");
            PlottingHelper.Plot(evidence, "Convergence of model evidence", string.Empty, "Iterations", "Evidence log odds");
        }

        public static void Online()
        {
            const int numBases = 128;
            const int batchSize = 1;

            // string filename = $"../../marginals/marginals_accel_full_{numBases}_bases.json";
            // var marginalsBatch = MarginalsCollection.Load(filename);

            // Learn in batches of 100
            var data = Data.LoadTrainTestData("accel", int.MaxValue, int.MaxValue, false);

            int numBatches = data.TrainSignals.RowCount/batchSize;
            // var bases = Enumerable.Repeat(numBases, numBatches).ToArray();

            int basisWidth = data.TrainSignals.ColumnCount;

            var results = new Results {Name = "online", Normalised = false};

            var models = ModelCollection.CreateModels();
            //                var priors = Marginals.CreatePriors(null, null, basisWidth, numBases, 1.0, false);
            var priors = Marginals.CreateHyperPriors(null, null, basisWidth, numBases);

            var testSet = data.TestSignals.SubMatrix(0, 200, 0, basisWidth);

            for (var i = 0; i < numBatches; i++)
            {
                string st = $"accel online {numBases} bases batch {i}";
                Console.WriteLine(st);

                var marginalsCollection = new MarginalsCollection {Name = st};

                var batch = data.TrainSignals.SubMatrix(i*batchSize, batchSize, 0, basisWidth);

                //                    Console.WriteLine($"Signal dimensions     ({batch.RowCount}, {batch.ColumnCount})");
                //                    Console.WriteLine($"Dictionary dimensions ({priors.DictionaryMeans.Length}, {priors.DictionaryMeans[0].Length})");
                //                    Console.WriteLine($"Dictionary dimensions ({priors.Dictionary.Length}, {priors.Dictionary[0].Length})");

                //                    marginalsCollection.TrainPosteriors = Models.TrainOnline.Train(hyperPriors, batch.ToRowArrays());
//                        marginalsCollection.TrainPosteriors = models.TrainOnline.Train(priors, batch.ToRowArrays());
                marginalsCollection.TrainPosteriors = models.Train.Train(priors, batch.ToRowArrays());

                // Print some information about the posteriors
                //                    foreach (var d in marginalsCollection.TrainPosteriors.Dictionary)
                //                    {
                //                        var means = d.GetMeans();
                //                        var vars = d.GetVariances();
                //                        Console.WriteLine($"means: [{means.Min()} - {means.Max()}]\tvariances: [{vars.Min()} - {vars.Max()}]");
                //                    }

                var means = marginalsCollection.TrainPosteriors.Dictionary.GetMeans<Gaussian>();
                var vars = marginalsCollection.TrainPosteriors.Dictionary.GetVariances<Gaussian>();
                Console.WriteLine(
                    $"means: [{means.Min(ia => ia.Min())} - {means.Max(ia => ia.Max())}]\tvariances: [{vars.Min(ia => ia.Min())} - {vars.Max(ia => ia.Max())}]");

                // Plot some bases
                if (false)
                {
                    PlottingHelper.PlotPosteriors(marginalsCollection.TrainPosteriors.DictionaryMeans, "Means",
                        st,
                        0,
                        16, 4, 4,
                        PlotType.Line, true);

                    PlottingHelper.PlotPosteriors(marginalsCollection.TrainPosteriors.DictionaryPrecisions,
                        "Precisions",
                        st, 0,
                        16, 4, 4,
                        PlotType.Line, true);
                }

                // Test using the current state
                if (i > 0 && i%10 == 0)
                {
                    // Get the cofficients for the test examples
//                            marginalsCollection.TestPosteriors =
//                                models.TrainFixedOnline.Train(marginalsCollection.TrainPosteriors, testSet.ToRowArrays());
                    marginalsCollection.TestPosteriors =
                        models.TrainFixed.Train(marginalsCollection.TrainPosteriors, testSet.ToRowArrays());

                    // Reconstruct
//                            marginalsCollection.Reconstructed = models.ReconstructOnline.Reconstruct(
//                                marginalsCollection.TestPosteriors).Signals;
                    marginalsCollection.Reconstructed = models.Reconstruct.Reconstruct(
                        marginalsCollection.TestPosteriors).Signals;

                    var reconstructions = data.TestSignals.ToRowArrays().Zip(marginalsCollection.Reconstructed,
                        (s, e) => new Reconstruction {Signal = s, Estimate = e}).ToArray();

                    // Compute average reconstruction error
                    results.Errors[i] = reconstructions.Select(ia => ia.Error).ToArray().Average();

                    string message = $"Reconstructions (online), avg. error={results.Errors[i]:N4}";
                    Console.WriteLine(message);
                }

                Console.WriteLine();

                priors = new Marginals
                {
                    Dictionary = marginalsCollection.TrainPosteriors.Dictionary.Copy()
                };

                //                    priors = new Marginals
                //                    {
                //                        Dictionary = null,
                //                        DictionaryMeans = marginalsCollection.TrainPosteriors.DictionaryMeans.Copy(),
                //                        DictionaryPrecisions =
                //                            marginalsCollection.TrainPosteriors.DictionaryPrecisions.Copy()
                //                    };
            }

            try
            {
                results.Save(Path.Combine(MainClass.ResultPath, data.Name));
            }
            catch (NullReferenceException e)
            {
                Console.WriteLine($"Failed to save results: {e.Message}");
            }
        }

        public static void Missing()
        {
            // Missing data experiments.
            // Take a trained model, and then test on data with more and more missing
            const int n = int.MaxValue;
            var bases = new[] {128};
            // var lambdas = Enumerable.Repeat(0.1, bases.Length).ToArray();

            const bool normalise = false;
            const bool sparse = false;
            const bool normConstraints = false;
            const bool useBias = false;

            var data = Data.LoadTrainTestData("accel", n, n, false, false);

            var resultsAccel = new ResultsCollection
            {
                Name = "missing",
                Normalised = normalise,
                NumTrain = data.TrainSignals.RowCount,
                NumTest = data.TestSignals.RowCount,
                Bases = bases
            };

            string st = $"nrm={normalise}, sparse={sparse}, nc={normConstraints}, bias={useBias}";
            Console.WriteLine(st);
            RunningExperiment.Models = ModelCollection.CreateModels(sparse, normConstraints, useBias);

            var priors = Marginals.CreatePriors(null, null, RunningExperiment.DataSet.TrainSignals.ColumnCount,
                bases[0], 1.0, false);
            RunningExperiment.Models.Train.Train(priors, RunningExperiment.DataSet.TrainSignals.ToRowArrays());

            // Compute the reconstruction error between the estimate and the uncorrupted truth
            // TODO: Need to be able to partially observe data

            // Plot a reconstruction

            resultsAccel.Save(MainClass.ResultPath);
        }


        public static void Mnist()
        {
            const int n = 100;
            // const int n = int.MaxValue;
            var bases = new[] {128};

            var data = Data.LoadTrainTestData("mnist", n, 100, false);

            // TODO: Posterior dictionary elements (bases) need to be
            // turned back into images
            // TODO: reconstruction

            var resultsMnist = new ResultsCollection
            {
                Name = data.Name,
                NumTrain = data.TrainSignals.RowCount,
                NumTest = data.TestSignals.RowCount,
                Bases = bases
            };

            var models = ModelCollection.CreateModels();
            resultsMnist.Results.Add(Run(data, ia => null, models, bases, "", false, true, false, false, false));
            resultsMnist.Save(MainClass.ResultPath);
        }


        /// <summary>
        /// Run the specified data, numSamples, bases, errors, evidence and subTitle.
        /// </summary>
        /// <param name="data">Data.</param>
        /// <param name="models">The model collection.</param>
        /// <param name="bases">Bases.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="precomputedFunction">Function that returns a <see cref="Precomputed"/> object.</param>
        /// <param name="isImage">Indicates whether the signals are 2D (images) or 1D</param>
        /// <param name="plotDictionary">Whether to plot the dictionary</param>
        /// <param name="plotCoefficients">Whether to plot the coefficients</param>
        /// <param name="plotReconstructions">Whether to plot the reconstructions</param>
        /// <param name="normalise">Whether to normalise the reconstructions.</param>
        private static Results Run(
            DataSet data,
            Func<int, Precomputed> precomputedFunction,
            ModelCollection models,
            IEnumerable<int> bases,
            string subTitle,
            bool normalise = true,
            bool isImage = false,
            bool plotDictionary = false,
            bool plotCoefficients = false,
            bool plotReconstructions = false)
        {
            var results = new Results {Name = subTitle, Normalised = normalise};

            foreach (int numBases in bases)
            {
                string n = normalise ? " (normalised)" : string.Empty;
                string st = $"{data.Name} {subTitle} {numBases} bases{n}";

                var precomputed = precomputedFunction(numBases);

                var priors = Marginals.CreateHyperPriors(precomputed?.Dictionary, precomputed?.TrainCoefficients,
                    data.TrainSignals.ColumnCount, numBases);

                Console.WriteLine($"Running {data.Name} {subTitle}, bases = {numBases}, normalise = {normalise}");
                Console.WriteLine(
                    $"Signal dimensions ({data.TrainSignals.RowCount}, {data.TrainSignals.ColumnCount})");
                Console.WriteLine($"Dictionary dimensions ({priors.Dictionary.Length}, {priors.Dictionary[0].Length})");

                var marginalsCollection = new MarginalsCollection {Name = st};

                if (models?.Train != null)
                {
                    if (models.Train.Parameters.Mode == Mode.TrainFixed)
                    {
                        marginalsCollection.TrainPosteriors = priors;
                    }
                    else
                    {
                        marginalsCollection.TrainPosteriors = models.Train.Train(priors, data.TrainSignals.ToRowArrays());
                        results.Evidence.Add(marginalsCollection.TrainPosteriors.Evidence.LogOdds);
                        Console.WriteLine($"Log Evidence: {marginalsCollection.TrainPosteriors.Evidence.LogOdds}");
                    }

                    // PlottingHelper.PlotResults(numBases, posteriors.DictionaryV, posteriors.CoefficientsV, "Toy data");


                    // TODO: Print dictionary norms and coefficient norms
                    Console.WriteLine($"Average dictionary norm: {marginalsCollection.AverageDictionaryNorm}");
                    Console.WriteLine($"Average coefficient norm: {marginalsCollection.AverageCoefficientNorm}");

                    // Get the cofficients for the test examples
                    marginalsCollection.TestPosteriors = models.TrainFixed.Train(
                        marginalsCollection.TrainPosteriors, data.TestSignals.ToRowArrays());

                    PlottingHelper.PlotResults(numBases, data.TrainSignals.ColumnCount,
                        marginalsCollection.TrainPosteriors.Dictionary, marginalsCollection.TrainPosteriors.Coefficients,
                        st, isImage, plotDictionary, plotCoefficients);

                    if (marginalsCollection.TrainPosteriors.Coefficients != null)
                    {
                        // Print sparsity
                        results.Sparsity.Add(marginalsCollection.AverageTrainSparsity);
                        Console.WriteLine($"Average sparsity: {marginalsCollection.AverageTrainSparsity}");
                    }

                    // Reconstruct
                    marginalsCollection.Reconstructed =
                        models.Reconstruct.Reconstruct(marginalsCollection.TestPosteriors).Signals;
                }
                else
                {
                    if (precomputed == null)
                    {
                        throw new InvalidOperationException("Precomputed dictionary required here");
                    }

                    results.Sparsity = precomputed.TrainCoefficients.EnumerateRows().Select(
                        ia => ia.Select(
                            ib => Math.Abs(ib) > double.Epsilon ? 0.0 : 1.0).Average()).ToArray();
                    Console.WriteLine($"Average sparsity: {results.Sparsity.Average()}");

                    var product = precomputed.TestCoefficients*precomputed.Dictionary;
                    marginalsCollection.Reconstructed = DistributionHelpers.GetGaussianArray(product.ToRowArrays());
                }

                if (normalise)
                {
                    marginalsCollection.Reconstructed = marginalsCollection.Reconstructed.Normalise();
                }

                // TODO: Save posteriors in more compact form
                // results.TrainPosteriors = trainPosteriors;
                // results.TestPosteriors = testPosteriors;
                // results.Reconstructed = reconstructed;

                var reconstructions = data.TestSignals.ToRowArrays().Zip(marginalsCollection.Reconstructed,
                    (s, e) => new Reconstruction {Signal = s, Estimate = e}).ToArray();

                // Compute average reconstruction error
                results.Errors.Add(reconstructions.Select(ia => ia.Error).ToArray().Average());
                Console.WriteLine($"Reconstruction error: {results.Errors.Last()}");

                // plot normalised reconstructions
                if (isImage)
                {
                    PlottingHelper.PlotImageReconstructions(reconstructions, results.Errors.Last(), 2, st, normalise,
                        plotReconstructions);
                }
                else
                {
                    PlottingHelper.PlotReconstructions(reconstructions, results.Errors.Last(), 2, 2, 1, st, normalise,
                        plotReconstructions);
                }

                try
                {
                    marginalsCollection.Save(MainClass.MarginalPath);
                }
                catch (NullReferenceException e)
                {
                    Console.WriteLine($"Failed to save results: {e.Message}");
                }
                Console.WriteLine();
            }

            results.Save(Path.Combine(MainClass.ResultPath, data.Name));
            return results;
        }
    }
}