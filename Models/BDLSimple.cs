//
// BDLSimple.cs
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

using System.Threading;

namespace BayesianDictionaryLearning.Models
{
    using System;
    using System.Linq;
    using MicrosoftResearch.Infer.Maths;
    using InferHelpers;
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;

    /// <summary>
    /// Bayesian Dictionary Learning
    /// </summary>
    public class BDLSimple : IModel
    {
        private Variable<int> numberOfSignals;
        private Variable<int> numberOfBases;
        private Variable<int> signalWidth;

        private VariableArray2D<double> coefficients;
        private VariableArray2D<double> dictionary;
        private Variable<double> noisePrecision;

        private VariableArray<double> bias;

        private VariableArray2D<double> signals;

        private Variable<bool> evidence;

//        private Variable<double> a;
//        private Variable<double> b;
//        private Variable<double> c;
//        private Variable<double> d;

//        private VariableArray2D<double> coefficientMeans;
        private VariableArray2D<double> coefficientPrecisions;
//        private VariableArray2D<double> dictionaryMeans;
        private VariableArray<VariableArray<double>, double[][]> dictionaryMeans;
        private VariableArray2D<double> dictionaryPrecisions;

        private VariableArray2D<Gaussian> coefficientPriors;
        private VariableArray2D<Gaussian> dictionaryPriors;

        private Variable<Gamma> noisePrecisionPrior;

        private VariableArray2D<bool> missing;

        private Range basis;
        private Range signal;
        private Range sample;

        private InferenceEngine engine;
        private IGeneratedAlgorithm compiledAlgorithm;

        public BDLParameters Parameters { get; set; }

        public bool Converged { get; set; }

        public InferenceMonitor InferenceMonitor { get; set; }

        public BDLSimple(BDLParameters parameters, bool autoConstruct = false)
            // Mode mode, bool nonNegative = false, bool sparse = true, bool normConstraints = false,
            // bool includeBias = false, bool showFactorGraph = false, bool autoConstruct = false, bool debug = true)
        {
            Parameters = parameters;
            if (autoConstruct)
            {
                ConstructModel();
            }
        }

        /// <summary>
        /// Constructs the model
        /// </summary>
        public void ConstructModel()
        {
            evidence = Variable.Bernoulli(0.5).Named("evidence");
            var evidenceBlock = Variable.If(evidence);

            numberOfBases = Variable.New<int>().Named("numberOfBases");
            numberOfSignals = Variable.New<int>().Named("numberOfSignals");
            signalWidth = Variable.New<int>().Named("signalWidth");

            basis = new Range(numberOfBases).Named("basis");
            signal = new Range(numberOfSignals).Named("signal");
            sample = new Range(signalWidth).Named("sample");

            // Hyperparameters
//            a = Variable.New<double>().Named("a").Attrib(new DoNotInfer());
//            b = Variable.New<double>().Named("b").Attrib(new DoNotInfer());
            var a = Variable.Observed(Parameters.Sparse ? 0.5 : 1.0).Named("a");
            var b = Variable.Observed(Parameters.Sparse ? 1e-6 : 1.0).Named("b");

            noisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior").Attrib(new DoNotInfer());
            noisePrecision = Variable<double>.Random(noisePrecisionPrior).Named("noisePrecision");

//            coefficientMeans = Variable.Array<double>(signal, basis).Named("coefficientMeans"); // .Attrib(new DoNotInfer());
            coefficientPrecisions = Variable.Array<double>(signal, basis).Named("coefficientPrecisions");
//            dictionaryMeans = Variable.Array<double>(basis, sample).Named("dictionaryMeans"); // .Attrib(new DoNotInfer());
            dictionaryMeans = Variable.Array(Variable.Array<double>(sample), basis).Named("dictionaryMeans");
                // .Attrib(new DoNotInfer());
            dictionaryPrecisions = Variable.Array<double>(basis, sample).Named("dictionaryPrecsions");
                // .Attrib(new DoNotInfer());

            coefficientPriors =
                Variable.Array<Gaussian>(signal, basis).Named("coefficentPriors").Attrib(new DoNotInfer());
            dictionaryPriors = Variable.Array<Gaussian>(basis, sample)
                .Named("dictionaryPriors")
                .Attrib(new DoNotInfer());

            // Define the arrays
            coefficients = Variable.Array<double>(signal, basis).Named("coefficients");
            dictionary = Variable.Array<double>(basis, sample).Named("dictionary");
            signals = Variable.Array<double>(signal, sample).Named("signals");

            signal.AddAttribute(new Sequential());

            // Priors

            // Coefficients and dictionary
//            coefficients[signal, basis] = Variable.GaussianFromMeanAndPrecision(coefficientMeans[signal, basis],
//                coefficientPrecisions[signal, basis]);

            switch (Parameters.Mode)
            {
                case Mode.Train:
                    // dictionaryMeans[basis, sample] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(basis, sample);
                    dictionaryMeans[basis][sample] = Variable.GaussianFromMeanAndVariance(0, 1).ForEach(basis, sample);
                    dictionaryPrecisions[basis, sample] = Variable.GammaFromShapeAndRate(1, 1).ForEach(basis, sample);
                    // dictionary[basis, sample] = Variable.GaussianFromMeanAndPrecision(dictionaryMeans[basis, sample], dictionaryPrecisions[basis, sample]);
                    dictionary[basis, sample] = Variable.GaussianFromMeanAndPrecision(dictionaryMeans[basis][sample],
                        dictionaryPrecisions[basis, sample]);
//                    dictionary[basis, sample] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(basis, sample);
                    coefficientPrecisions[signal, basis] = Variable.GammaFromShapeAndRate(a, b).ForEach(signal, basis);
                    coefficients[signal, basis] = Variable.GaussianFromMeanAndPrecision(0,
                        coefficientPrecisions[signal, basis]);
                    break;
                case Mode.TrainFixed:
                    dictionary[basis, sample] = Variable.Random<double, Gaussian>(dictionaryPriors[basis, sample]);
                    coefficientPrecisions[signal, basis] = Variable.GammaFromShapeAndRate(a, b).ForEach(signal, basis);
                    coefficients[signal, basis] = Variable.GaussianFromMeanAndPrecision(0,
                        coefficientPrecisions[signal, basis]);
                    dictionary.AddAttribute(new DoNotInfer());
                    break;
//                case Mode.TrainOnline:
//                    dictionary[basis, sample] = Variable.Random<double, Gaussian>(dictionaryPriors[basis, sample]);
//                    coefficientPrecisions[signal, basis] = Variable.GammaFromShapeAndRate(a, b).ForEach(signal, basis);
//                    coefficients[signal, basis] = Variable.GaussianFromMeanAndPrecision(0,
//                        coefficientPrecisions[signal, basis]);
//                    break;
                case Mode.Reconstruct:
                    dictionary[basis, sample] = Variable.Random<double, Gaussian>(dictionaryPriors[basis, sample]);
                    coefficients[signal, basis] = Variable.Random<double, Gaussian>(coefficientPriors[signal, basis]);
                    dictionary.AddAttribute(new DoNotInfer());
                    coefficients.AddAttribute(new DoNotInfer());
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(Mode));
            }

            // Break symmetry
//            coefficients[signal, basis].InitialiseTo(coefficientPriors[signal, basis]);
            dictionary[basis, sample].InitialiseTo(dictionaryPriors[basis, sample]);

            // The main model
            VariableArray2D<double> cleanSignals;
            if (Parameters.UseMatrixMultiply)
            {
                if (Parameters.MissingData)
                {
                    throw new InvalidOperationException("Cannot use MatrixMultiply factor with missing data");
                }

                cleanSignals = Variable.MatrixMultiply(coefficients, dictionary).Named("clean");
            }
            else
            {
                if (Parameters.MissingData)
                {
                    missing = Variable.Array<bool>(signal, sample).Named("missing").Attrib(new DoNotInfer());
                    cleanSignals = Helpers.MatrixMultiply(coefficients, dictionary, signal, sample, basis, missing).Named("clean");
                }
                else
                {
                    cleanSignals = Helpers.MatrixMultiply(coefficients, dictionary, signal, sample, basis).Named("clean");
                }
            }

            if (Parameters.IncludeBias)
            {
                bias = Variable.Array<double>(sample).Named("bias");
                bias[sample] = Variable.GaussianFromMeanAndPrecision(0.0, 0.01).ForEach(sample);
                var signalPlusBias = Variable.Array<double>(signal, sample).Named("signalPlusBias");
                signalPlusBias[signal, sample] = cleanSignals[signal, sample] + bias[sample];
                signals[signal, sample] = Variable.GaussianFromMeanAndPrecision(signalPlusBias[signal, sample],
                    noisePrecision);
            }
            else
            {
                signals[signal, sample] = Variable.GaussianFromMeanAndPrecision(cleanSignals[signal, sample],
                    noisePrecision);
            }

            if (Parameters.NonNegative)
            {
                Variable.ConstrainPositive(coefficients[signal, basis]);
            }

            if (Parameters.Mode == Mode.Train && Parameters.NormConstraints)
            {
                Helpers.ConstrainNorms(dictionaryMeans, basis, sample);
            }

            evidenceBlock.CloseBlock();

            InitialiseEngine();
        }

        /// <summary>
        /// Initialises the engine.
        /// </summary>
        /// <returns>The engine.</returns>
        public void InitialiseEngine()
        {
            engine = new InferenceEngine
            {
                Algorithm = new VariationalMessagePassing(),
                ShowFactorGraph = Parameters.Mode == Mode.Train && Parameters.ShowFactorGraph,
                ModelName = "BayesianDictionaryLearning"
            };

            switch (Parameters.Mode)
            {
                case Mode.Train:
//                case Mode.TrainOnline:
                    engine.OptimiseForVariables = new IVariable[] {dictionary, coefficients, noisePrecision, evidence};
                    break;
                case Mode.TrainFixed:
                    engine.OptimiseForVariables = new IVariable[] {coefficients, noisePrecision, evidence};
                    break;
                case Mode.Reconstruct:
                    engine.OptimiseForVariables = new IVariable[] {signals, evidence};
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(Mode));
            }

            if (Parameters.Debug)
            {
                engine.ShowTimings = true;
                engine.Compiler.IncludeDebugInformation = true;
                engine.Compiler.GenerateInMemory = false;
                engine.Compiler.GeneratedSourceFolder = "../../GeneratedSource";
            }
            else
            {
                // Speed ups
                engine.ShowWarnings = false;
                engine.Compiler.ReturnCopies = false;
                engine.Compiler.FreeMemory = false;
                engine.Compiler.CatchExceptions = false;
                engine.Compiler.UseParallelForLoops = true;
                engine.Compiler.WriteSourceFiles = false;
                engine.Compiler.AddComments = false;
            }

            numberOfBases.ObservedValue = default(int);
            numberOfSignals.ObservedValue = default(int);
            signalWidth.ObservedValue = default(int);
//            a.ObservedValue = default(double);
//            b.ObservedValue = default(double);
            noisePrecisionPrior.ObservedValue = default(Gamma);

            if (Parameters.MissingData)
            {
                missing.ObservedValue = new bool[,] {};
            }

            if (Parameters.Mode == Mode.Train || Parameters.Mode == Mode.TrainFixed) //  || Parameters.Mode == Mode.TrainOnline)
            {
                signals.ObservedValue = new double[,] {};
            }
            else
            {
                coefficientPrecisions.ObservedValue = new double[,] {};
            }

            dictionaryPriors.ObservedValue = new Gaussian[,] {};
            coefficientPriors.ObservedValue = new Gaussian[,] {};

            compiledAlgorithm = engine.GetCompiledInferenceAlgorithm(engine.OptimiseForVariables.ToArray());
            compiledAlgorithm.ProgressChanged += StandardHandler;
        }

        /// <summary>
        /// Train the model.
        /// </summary>
        /// <param name="priors">The priors.</param>
        /// <param name="inputSignals">The input signals.</param>
        public Marginals Train(Marginals priors, double[][] inputSignals)
        {
            if (priors == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors));
            }

            if (priors.Dictionary == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors.Dictionary));
            }

            if (inputSignals == null)
            {
                throw new ArgumentOutOfRangeException(nameof(inputSignals));
            }

            int numSignals = inputSignals.Length;
            int numBases = priors.Dictionary.Length;
            int obsSignalWidth = inputSignals[0].Length;

            // Assume that first signal is the correct width
            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            compiledAlgorithm.SetObservedValue(noisePrecisionPrior.NameInGeneratedCode, Gamma.FromShapeAndRate(1, 1));
            // compiledAlgorithm.SetObservedValue(coefficientMeans.NameInGeneratedCode, ArrayHelpers.Zeros(numBases, numSignals).To2D());
            // coefficientPrecisions.InitialiseTo(ArrayHelpers.Zeros(numBases, numSignals));
            //coefficientPrecisions.InitialiseTo(
            //    Distribution<double>.Array(DistributionHelpers.CreateGammaArray(numBases, numSignals, 1.0, 1.0)));

            compiledAlgorithm.SetObservedValue(signals.NameInGeneratedCode, inputSignals.To2D());

            if (Parameters.MissingData)
            {
                compiledAlgorithm.SetObservedValue(missing.NameInGeneratedCode,
                    inputSignals.Select(ia => ia.Select(double.IsNaN).ToArray()).ToArray().To2D());
            }

            if (Parameters.Mode == Mode.Train)
            {
                Rand.Restart(0);
                compiledAlgorithm.SetObservedValue(dictionaryPriors.NameInGeneratedCode,
                    DistributionHelpers.CreateGaussianArray(numBases, obsSignalWidth, Rand.Normal, 1.0).To2D());
            }
            else
            {
                compiledAlgorithm.SetObservedValue(dictionaryPriors.NameInGeneratedCode, priors.Dictionary.To2D());
            }

//            compiledAlgorithm.SetObservedValue(coefficientPriors.NameInGeneratedCode,
//                DistributionHelpers.CreateGaussianArray(numSignals, numBases, Rand.Normal, 1.0).To2D());


            RunInference();
            return GetCurrentMarginals(priors);
        }

        /// <summary>
        /// Reconstruct using the dictionary and coefficients provided.
        /// </summary>
        /// <param name="priors">The priors.</param>
        public Marginals Reconstruct(Marginals priors)
        {
            if (priors == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors));
            }

            if (priors.Dictionary == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors.Dictionary));
            }

            if (priors.Coefficients == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors.Coefficients));
            }

            int numSignals = priors.Coefficients.Length;
            int numBases = priors.Dictionary.Length;
            int obsSignalWidth = priors.Dictionary[0].Length;

            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            compiledAlgorithm.SetObservedValue(dictionaryPriors.NameInGeneratedCode, priors.Dictionary.To2D());
//            compiledAlgorithm.SetObservedValue(coefficientMeans.NameInGeneratedCode, priors.Coefficients.GetMeans<Gaussian>().To2D());
//            compiledAlgorithm.SetObservedValue(coefficientPrecisions.NameInGeneratedCode, priors.Coefficients.GetPrecisions().To2D());
            compiledAlgorithm.SetObservedValue(coefficientPriors.NameInGeneratedCode, priors.Coefficients.To2D());
            compiledAlgorithm.SetObservedValue(noisePrecisionPrior.NameInGeneratedCode, priors.NoisePrecision);

            if (Parameters.MissingData)
            {
                compiledAlgorithm.SetObservedValue(missing.NameInGeneratedCode,
                    ArrayHelpers.Uniform(numSignals, obsSignalWidth, false).To2D());
            }

            RunInference();
            return GetCurrentMarginals(null);
        }

        /// <summary>
        /// Set observed values
        /// </summary>
        /// <param name="numSignals">The number of signals.</param>
        /// <param name="obsSignalWidth">The width of each signal.</param>
        /// <param name="numBases">The number of bases.</param>
        private void SetObservedVariables(int numSignals, int obsSignalWidth, int numBases)
        {
            // Assume that first signal is the correct width
            compiledAlgorithm.SetObservedValue(numberOfSignals.NameInGeneratedCode, numSignals);
            compiledAlgorithm.SetObservedValue(signalWidth.NameInGeneratedCode, obsSignalWidth);
            compiledAlgorithm.SetObservedValue(numberOfBases.NameInGeneratedCode, numBases);
//            compiledAlgorithm.SetObservedValue(a.NameInGeneratedCode, 1.0); // 0.5);
//            compiledAlgorithm.SetObservedValue(b.NameInGeneratedCode, 1.0); // 1e-6);
//            if (Mode == Mode.Train)
//            {
//                compiledAlgorithm.SetObservedValue(c.NameInGeneratedCode, 1.0);
//                compiledAlgorithm.SetObservedValue(d.NameInGeneratedCode, 1.0);
//            }
        }

        private void RunInference()
        {
            Runner.RunningExperiment.RunningModel = this;
            Converged = false;
            InferenceMonitor = new InferenceMonitor();
            compiledAlgorithm.Reset();
            for (var i = 0; i < Parameters.MaxIterations[Parameters.Mode]; i++)
            {
                try
                {
                    compiledAlgorithm.Update(1);
                }
                catch (NotSupportedException exception)
                {
                    Console.WriteLine($"Inference failed with exception: {exception.Message}");
                    return;
                }

                if (Converged)
                {
                    return;
                }
            }
        }

        private void StandardHandler(object sender, ProgressChangedEventArgs eventArgs)
        {
            InferenceMonitor.PreviousEvidence = InferenceMonitor.CurrentEvidence;
            InferenceMonitor.CurrentEvidence = compiledAlgorithm.Marginal<Bernoulli>("evidence").LogOdds;

            if (eventArgs.Iteration <= 0)
            {
                return;
            }

            if (Parameters.ShowProgress)
            {
//                Console.Write(eventArgs.Iteration%10 == 0 ? $"{eventArgs.Iteration}" : ".");
                Console.WriteLine($"{Parameters.Mode} {eventArgs.Iteration}: {InferenceMonitor.EvidenceRatio}");
            }

//            if (InferenceMonitor.MeanDifference >= Tolerance)
            // Check for convergence
            if (!Parameters.ConvergenceCriterion(InferenceMonitor, Parameters.Tolerance))
            {
                return;
            }

            if (Parameters.ShowProgress)
            {
                // Console.WriteLine($"\n{MarginalOfInterest} converged after {eventArgs.Iteration + 1} iterations");
                Console.WriteLine($"{Parameters.Mode} converged after {eventArgs.Iteration + 1} iterations");
                Console.WriteLine();
            }

            Converged = true;
        }

        /// <summary>
        /// Add a handler for when an inference update happens.
        /// </summary>
        /// <param name="handler">The event handler.</param>
        public void AddUpdateHandler(EventHandler<ProgressChangedEventArgs> handler)
        {
            System.Diagnostics.Debug.Assert(compiledAlgorithm != null, "compiledAlgorithm != null");
            compiledAlgorithm.ProgressChanged += handler;
        }

        public Marginals GetCurrentMarginals(Marginals priors)
        {
            if (Parameters.Mode == Mode.Reconstruct)
            {
                return new Marginals
                {
                    Signals = compiledAlgorithm.Marginal<Gaussian[,]>(signals.NameInGeneratedCode).ToJagged()
                };
            }

            var posteriors = new Marginals
            {
                Coefficients = compiledAlgorithm.Marginal<Gaussian[,]>(coefficients.NameInGeneratedCode).ToJagged(),
                NoisePrecision = compiledAlgorithm.Marginal<Gamma>(noisePrecision.NameInGeneratedCode),
                Evidence = compiledAlgorithm.Marginal<Bernoulli>(evidence.NameInGeneratedCode)
            };

            switch (Parameters.Mode)
            {
                case Mode.Train:
//                case Mode.TrainOnline:
                    posteriors.Dictionary =
                        compiledAlgorithm.Marginal<Gaussian[,]>(dictionary.NameInGeneratedCode).ToJagged();
                    break;
                case Mode.TrainFixed:
                    posteriors.Dictionary = priors.Dictionary.Copy();
                    break;
                case Mode.Reconstruct:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            return posteriors;
        }

        public override string ToString()
        {
            return $"BDL ({Parameters.Mode})";
        }
    }

    public class InferenceMonitor
    {
        public double PreviousEvidence { get; set; } = double.NegativeInfinity;
        public double CurrentEvidence { get; set; }

        public double EvidenceRatio => CurrentEvidence / PreviousEvidence;
    }
}

