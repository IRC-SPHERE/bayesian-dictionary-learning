//
// BDL.cs
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

namespace BayesianDictionaryLearning.Models
{
    using System;
    using System.Collections.Generic;
    using InferHelpers;
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Models;

    /// <summary>
    /// Bayesian Dictionary Learning (Gaussian version)
    /// </summary>
    public class BDL : IModel
    {
        private Variable<int> numberOfSignals;
        private Variable<int> numberOfBases;
        private Variable<int> signalWidth;

        private VariableArray2D<double> coefficients;
        private VariableArray2D<double> dictionary;

        private VariableArray2D<double> signals;

        private Variable<bool> evidence;

//        private Variable<double> beta;
        private Variable<double> a;
        private Variable<double> b;

        private VariableArray<VariableArray <double>, double[][]> coefficientMeans;
        private VariableArray<VariableArray <double>, double[][]> coefficientPrecisions;

        private VariableArray<VariableArray <double>, double[][]> dictionaryMeans;
        private VariableArray<VariableArray <double>, double[][]> dictionaryPrecisions;

        private Variable<double> noisePrecision;
        private Variable<Gamma> noisePrecisionPrior;

        private Range basis;
        private Range signal;
        private Range sample;

        private InferenceEngine engine;

        public BDLParameters Parameters { get; set; }

        public BDL(BDLParameters parameters, bool autoConstruct = false)
        {
            Parameters = parameters;
            if (autoConstruct)
            {
                ConstructModel();
            }
        }

        /// <summary>
        /// Add a handler for when an inference update happens.
        /// </summary>
        /// <param name="handler">The event handler.</param>
        public void AddUpdateHandler(EventHandler<ProgressChangedEventArgs> handler)
        {
            throw new NotImplementedException();
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
            a = Variable.New<double>().Named("a"); //.Attrib(new DoNotInfer());
            b = Variable.New<double>().Named("b"); //.Attrib(new DoNotInfer());

            // beta = Variable.New<double>().Named("beta"); // .Attrib(new DoNotInfer());
//            beta = Variable.GammaFromShapeAndRate(1, 1).Named("beta");

            noisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior"); //.Attrib(new DoNotInfer());
            noisePrecision = Variable<double>.Random(noisePrecisionPrior).Named("noisePrecision");

            coefficientMeans = Variable.Array(Variable.Array<double>(basis), signal).Named("coefficientMeans"); // .Attrib(new DoNotInfer());
            coefficientPrecisions = Variable.Array(Variable.Array<double>(basis), signal).Named("coefficientPrecisions");

            dictionaryMeans = Variable.Array(Variable.Array<double>(sample), basis).Named("dictionaryMeans"); // .Attrib(new DoNotInfer());
            dictionaryPrecisions = Variable.Array(Variable.Array<double>(sample), basis).Named("dictionaryPrecisions"); // .Attrib(new DoNotInfer());

            // Define the arrays
            coefficients = Variable.Array<double>(signal, basis).Named("coefficients");
            dictionary = Variable.Array<double>(basis, sample).Named("dictionary");
            signals = Variable.Array<double>(signal, sample).Named("signals");

            signal.AddAttribute(new Sequential());

            // Priors
            coefficientPrecisions[signal][basis] = Variable.GammaFromShapeAndRate(a, b).ForEach(signal, basis);

            using (Variable.ForEach(basis))
            {
                using (Variable.ForEach(sample))
                {
                    dictionaryMeans[basis][sample] = Variable.GaussianFromMeanAndPrecision(0, 1);
                    // dictionaryPrecisions[basis][sample].SetTo(beta);
                    dictionaryPrecisions[basis][sample] = Variable.GammaFromShapeAndRate(1, 1);
                }
            }

            // Coefficients and dictionary
            coefficients[signal, basis] = Variable.GaussianFromMeanAndPrecision(coefficientMeans[signal][basis], coefficientPrecisions[signal][basis]);
            dictionary[basis, sample] = Variable.GaussianFromMeanAndPrecision(dictionaryMeans[basis][sample], dictionaryPrecisions[basis][sample]);

            // The main model
            var cleanSignals = Variable.MatrixMultiply(coefficients, dictionary).Named("clean");
            //var cleanSignals = MatrixMultiply(coefficients, dictionary).Named("clean");
            signals[signal, sample] = Variable.GaussianFromMeanAndPrecision(cleanSignals[signal, sample], noisePrecision);

            // Norm and Non-negativity constraints
            ConstrainNorms();

            if (Parameters.NonNegative)
            {
                // TODO: not sure how to do this with vectors?
                //Variable.ConstrainPositive(coefficients[signal, basis]);
                //Variable.ConstrainPositive(coefficients[signal][basis]);
            }

            if (Parameters.Mode == Mode.TrainFixed)
            {
                dictionary.AddAttribute(new DoNotInfer());
            }

            evidenceBlock.CloseBlock();
            InitialiseEngine();
        }

        /// <summary>
        /// Constrains the norms.
        /// </summary>
        /// <returns>The norms.</returns>
        public void ConstrainNorms()
        {
//            var squares = Variable.Array<double>(basis).Named("squares");
//            var clone = sample.Clone().Named("clone");
//
//            using (Variable.ForEach(basis))
//            {
//                var v1 = Variable.Array<double>(sample).Named("v1");
//                using (Variable.ForEach(sample))
//                {
//                    v1[sample] = dictionary[basis, sample];
//                }
//
//                var v2 = Variable.Array<double>(clone).Named("v1");
//                using (Variable.ForEach(clone))
//                {
//                    v1[clone] = Variable.Copy(dictionary[basis, clone]);
//                }
//
//                squares[basis] = Variable.InnerProduct(Variable.Vector(v1), Variable.Vector(v2));
//                Variable.ConstrainEqual(squares[basis], 1.0);
//            }

//            var clone = basis.Clone().Named("clone");
//            var copy = Variable.Array<double>(sample, clone).Named("copy");
//            copy[sample, basis] = Variable.Copy(dictionary[basis, sample]);
//            var cov = Variable.MatrixMultiply(dictionary, copy).Named("cov");
//
//            using (Variable.ForEach(basis))
//            {
//                using (Variable.ForEach(clone))
//                {
//                    Variable.ConstrainEqual(cov[basis, clone], 1.0);
//                }
//            }

            //using (Variable.ForEach(basis))
            //{
            //    using (Variable.ForEach(sample))
            //    {
            //        Variable.ConstrainBetween(dictionary[basis, sample], -1, 1);
            //    }
            //}

            // TODO: Can this be done for this version?
            //using (Variable.ForEach(sample))
            //{
            //    var copy = Variable.Copy(dictionary[sample]).Named("dCopy");
            //    var normSquared = Variable.InnerProduct(dictionary[sample], copy).Named("normSquared");
            //    //Variable.ConstrainPositive(1 - normSquared);
            //    //Variable.ConstrainEqual(normSquared, 1.0);
            //    Variable.ConstrainEqualRandom(normSquared, Gaussian.FromMeanAndVariance(1.0, 0.1));
            //}
        }

        /// <summary>
        /// Initialises the engine.
        /// </summary>
        /// <returns>The engine.</returns>
        public void InitialiseEngine()
        {
            engine = new InferenceEngine { Algorithm = new VariationalMessagePassing() };
            engine.Compiler.IncludeDebugInformation = true;
            // engine.ShowFactorGraph = Mode == Mode.Train && ShowFactorGraph;
            engine.ShowFactorGraph = Parameters.ShowFactorGraph;

            engine.Algorithm.DefaultNumberOfIterations = Parameters.MaxIterations[Parameters.Mode];

            switch (Parameters.Mode)
            {
                case Mode.Train:
                    engine.OptimiseForVariables = new IVariable[]
                    {
                        dictionary, dictionaryMeans, dictionaryPrecisions,
                        coefficients, noisePrecision, evidence
                    };
                    break;
                case Mode.TrainFixed:
                    engine.OptimiseForVariables = new IVariable[] { coefficients, noisePrecision, evidence };
                break;
                case Mode.Reconstruct:
                    engine.OptimiseForVariables = new IVariable[] { signals, evidence };
                break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(Mode));
            }

            // Speed ups
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = false;
            engine.Compiler.CatchExceptions = false;
            engine.Compiler.UseParallelForLoops = true;
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
            int numBases = priors.Dictionary?.Length ?? priors.DictionaryMeans.Length;
            int obsSignalWidth = inputSignals[0].Length;

            // Assume that first signal is the correct width
            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            noisePrecisionPrior.ObservedValue = Gamma.FromShapeAndRate(1, 1);
            coefficientMeans.ObservedValue = ArrayHelpers.Zeros(numBases, numSignals);

            signals.ObservedValue = inputSignals.To2D();

            if (Parameters.Mode == Mode.TrainFixed)
            {
                dictionaryMeans.ObservedValue = priors.Dictionary.GetMeans<Gaussian>();
                dictionaryPrecisions.ObservedValue = priors.Dictionary.GetPrecisions();
            }
            else
            {
                // Break symmetry with random initialisation of the dictionary
                dictionary.InitialiseTo(Distribution<double>.Array(priors.Dictionary.To2D()));
            }

            // TODO: Try spike and slab
            // TODO: Try constraining the norm of the bases somehow
            var posteriors = new Marginals
            {
                Coefficients = engine.Infer<Gaussian[,]>(coefficients).ToJagged(),
                NoisePrecision = engine.Infer<Gamma>(noisePrecision),
                Evidence = engine.Infer<Bernoulli>(evidence)
            };

            switch (Parameters.Mode)
            {
                case Mode.Train:
                    posteriors.Dictionary = engine.Infer<Gaussian[,]>(dictionary).ToJagged();
                    posteriors.DictionaryMeans = engine.Infer<Gaussian[][]>(dictionaryMeans);
                    posteriors.DictionaryPrecisions = engine.Infer<Gamma[][]>(dictionaryPrecisions);
                    break;
                case Mode.TrainFixed:
                    posteriors.Dictionary = priors.Dictionary;
                    break;
                case Mode.Reconstruct:
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }

            return posteriors;
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

            if (Parameters.Mode == Mode.Reconstruct)
            {
                if (priors.Dictionary == null)
                {
                    throw new ArgumentOutOfRangeException(nameof(priors.Dictionary));
                }
            }
            else
            {
                if (priors.DictionaryMeans == null)
                {
                    throw new ArgumentOutOfRangeException(nameof(priors.DictionaryMeans));
                }

                if (priors.DictionaryPrecisions == null)
                {
                    throw new ArgumentOutOfRangeException(nameof(priors.DictionaryPrecisions));
                }
            }

            if (priors.Coefficients == null)
            {
                throw new ArgumentOutOfRangeException(nameof(priors.Coefficients));
            }

            int numSignals = priors.Coefficients.Length;
            int numBases = priors.Dictionary?.Length ?? priors.DictionaryMeans.Length;
            int obsSignalWidth = priors.Dictionary?[0].Length ?? priors.DictionaryMeans[0].Length;

            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            if (Parameters.Mode == Mode.Reconstruct)
            {
                dictionaryMeans.ObservedValue = priors.Dictionary.GetMeans<Gaussian>();
                dictionaryPrecisions.ObservedValue = priors.Dictionary.GetPrecisions();
            }

            coefficientMeans.ObservedValue = priors.Coefficients.GetMeans<Gaussian>();
            coefficientPrecisions.ObservedValue = priors.Coefficients.GetPrecisions();

            noisePrecisionPrior.ObservedValue = priors.NoisePrecision;

            return new Marginals
            {
                Signals = engine.Infer<Gaussian[,]>(signals).ToJagged()
            };
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
            numberOfSignals.ObservedValue = numSignals;
            signalWidth.ObservedValue = obsSignalWidth;
            numberOfBases.ObservedValue = numBases;

            a.ObservedValue = 0.5;
            b.ObservedValue = 1e-6;

            // TODO: Try the two different priors for beta here
            // beta.ObservedValue = 1;
            //beta.ObservedValue = 1e-8;
            //Console.WriteLine($"Using sigma={sigma:N2}, bound={bound:N2}, beta={beta.ObservedValue:N2}");
        }
    }
}

