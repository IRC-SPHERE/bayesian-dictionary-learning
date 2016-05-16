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
    using System.Linq;

    using InferHelpers;
    using MicrosoftResearch.Infer;
    using MicrosoftResearch.Infer.Distributions;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;

    /// <summary>
    /// Bayesian Dictionary Learning (Vector Gaussian version)
    /// </summary>
    public class BDLV
    {
        private Variable<int> numberOfSignals;
        private Variable<int> numberOfBases;
        private Variable<int> signalWidth;

        private VariableArray<Vector> coefficients;
        private VariableArray<Vector> dictionary;

        private VariableArray2D<double> signals;

        private Variable<bool> evidence;

        private Variable<PositiveDefiniteMatrix> beta;
        private Variable<double> a;
        private Variable<PositiveDefiniteMatrix> b;

        private VariableArray<Vector> coefficientMeans;
        private VariableArray<PositiveDefiniteMatrix> coefficientPrecisions;

        private VariableArray<Vector> dictionaryMeans;
        private VariableArray<PositiveDefiniteMatrix> dictionaryPrecisions;

        private Variable<double> noisePrecision;
        private Variable<Gamma> noisePrecisionPrior;
        //private Range basis;
        private Range signal;
        private Range sample;

        private Variable<Vector> zero;

        private InferenceEngine engine;

        public BDLParameters Parameters { get; set; }

        public BDLV(BDLParameters parameters, bool autoConstruct = false)
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

            //basis = new Range(numberOfBases).Named("basis");
            signal = new Range(numberOfSignals).Named("signal");
            sample = new Range(signalWidth).Named("sample");

            zero = Variable.New<Vector>().Named("zero").Attrib(new DoNotInfer());

            // Hyperparameters
            a = Variable.New<double>().Named("a").Attrib(new DoNotInfer());
            b = Variable.New<PositiveDefiniteMatrix>().Named("b").Attrib(new DoNotInfer());

            beta = Variable.New<PositiveDefiniteMatrix>().Named("beta").Attrib(new DoNotInfer());

            noisePrecisionPrior = Variable.New<Gamma>().Named("noisePrecisionPrior").Attrib(new DoNotInfer());
            noisePrecision = Variable<double>.Random(noisePrecisionPrior).Named("noisePrecision");

            coefficientMeans = Variable.Array<Vector>(signal).Named("coefficientMeans").Attrib(new DoNotInfer());
            coefficientPrecisions = Variable.Array<PositiveDefiniteMatrix>(signal).Named("coefficientPrecisions");

            dictionaryMeans = Variable.Array<Vector>(sample).Named("dictionaryMeans"); // .Attrib(new DoNotInfer());
            dictionaryPrecisions = Variable.Array<PositiveDefiniteMatrix>(sample).Named("dictionaryPrecisions"); // .Attrib(new DoNotInfer());

            // Define the arrays
            coefficients = Variable.Array<Vector>(signal).Named("coefficients");
            dictionary = Variable.Array<Vector>(sample).Named("dictionary");
            signals = Variable.Array<double>(signal, sample).Named("signals");

            signal.AddAttribute(new Sequential());

            // Priors
            coefficientPrecisions[signal] = Variable.WishartFromShapeAndRate(a, b).ForEach(signal);

            dictionaryMeans[sample] = zero;
            dictionaryPrecisions[sample].SetTo(beta);

            // Coefficients and dictionary
            coefficients[signal] = Variable.VectorGaussianFromMeanAndPrecision(coefficientMeans[signal], coefficientPrecisions[signal]);
            dictionary[sample] = Variable.VectorGaussianFromMeanAndPrecision(dictionaryMeans[sample], dictionaryPrecisions[sample]);

            // The main model
            //var cleanSignals = Variable.MatrixMultiply(coefficients, dictionary).Named("clean");
            var cleanSignals = Helpers.MatrixMultiply(coefficients, dictionary, signal, sample).Named("clean");
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
            using (Variable.ForEach(sample))
            {
                var copy = Variable.Copy(dictionary[sample]).Named("dCopy");
                var normSquared = Variable.InnerProduct(dictionary[sample], copy).Named("normSquared");
                //Variable.ConstrainPositive(1 - normSquared);
                //Variable.ConstrainEqual(normSquared, 1.0);
                Variable.ConstrainEqualRandom(normSquared, Gaussian.FromMeanAndVariance(1.0, 0.1));
            }
        }

        /// <summary>
        /// Initialises the engine.
        /// </summary>
        /// <returns>The engine.</returns>
        public void InitialiseEngine()
        {
            engine = new InferenceEngine { Algorithm = new VariationalMessagePassing() };
            engine.Compiler.IncludeDebugInformation = true;
            engine.ShowFactorGraph = Parameters.Mode == Mode.Train && Parameters.ShowFactorGraph;

            switch (Parameters.Mode)
            {
                case Mode.Train:
                    engine.Algorithm.DefaultNumberOfIterations = 20;
                    engine.OptimiseForVariables = new IVariable[] { dictionary, coefficients, noisePrecision, evidence };
                break;
                case Mode.TrainFixed:
                    engine.Algorithm.DefaultNumberOfIterations = 5;
                    engine.OptimiseForVariables = new IVariable[] { coefficients, noisePrecision, evidence };
                break;
                case Mode.Reconstruct:
                    engine.Algorithm.DefaultNumberOfIterations = 2;
                    engine.OptimiseForVariables = new IVariable[] { signals, evidence };
                break;
            }

            // Speed ups
            engine.Compiler.ReturnCopies = false;
            engine.Compiler.FreeMemory = true;
            engine.Compiler.CatchExceptions = false;
        }

        /// <summary>
        /// Train the model.
        /// </summary>
        /// <param name="priors">The priors.</param>
        /// <param name="inputSignals">The input signals.</param>
        public Marginals Train(Marginals priors, double[][] inputSignals)
        {
            if (priors == null)
                throw new System.ArgumentOutOfRangeException(nameof(priors));

            if (priors.DictionaryV == null)
                throw new System.ArgumentOutOfRangeException(nameof(priors.DictionaryV));

            if (inputSignals == null)
                throw new System.ArgumentOutOfRangeException(nameof(inputSignals));

            int numSignals = inputSignals.Length;
            int numBases = priors.DictionaryV[0].Dimension;
            int obsSignalWidth = inputSignals[0].Length;

            // Assume that first signal is the correct width
            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            noisePrecisionPrior.ObservedValue = Gamma.FromShapeAndRate(1, 1);
            coefficientMeans.ObservedValue = Enumerable.Repeat(Vector.Zero(numBases), numSignals).ToArray();

            this.signals.ObservedValue = inputSignals.To2D();

            // Break symmetry with random initialisation of the dictionary
            dictionary.InitialiseTo(Distribution<Vector>.Array(priors.DictionaryV));

            if (Parameters.Mode == Mode.TrainFixed)
            {
                dictionaryMeans.ObservedValue = priors.DictionaryV.GetMeans();
                dictionaryPrecisions.ObservedValue = priors.DictionaryV.GetPrecisions();
            }

            // TODO: Try spike and slab
            // TODO: Try constraining the norm of the bases somehow
            var posteriors = new Marginals
            {
                CoefficientsV = engine.Infer<VectorGaussian[]>(coefficients),
                NoisePrecision = engine.Infer<Gamma>(noisePrecision),
                Evidence = engine.Infer<Bernoulli>(evidence)
            };

            if (Parameters.Mode == Mode.Train)
            {
                posteriors.DictionaryV = engine.Infer<VectorGaussian[]>(dictionary);
            }
            else
            {
                posteriors.DictionaryV = priors.DictionaryV;
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
                throw new System.ArgumentOutOfRangeException(nameof(priors));

            if (priors.DictionaryV == null)
                throw new System.ArgumentOutOfRangeException(nameof(priors.DictionaryV));

            if (priors.CoefficientsV == null)
                throw new System.ArgumentOutOfRangeException(nameof(priors.CoefficientsV));

            int numSignals = priors.CoefficientsV.Length;
            int numBases = priors.DictionaryV[0].Dimension;
            int obsSignalWidth = priors.DictionaryV.Length;

            SetObservedVariables(numSignals, obsSignalWidth, numBases);

            dictionaryMeans.ObservedValue = priors.DictionaryV.GetMeans();
            dictionaryPrecisions.ObservedValue = priors.DictionaryV.GetPrecisions();

            coefficientMeans.ObservedValue = priors.CoefficientsV.GetMeans();
            coefficientPrecisions.ObservedValue = priors.CoefficientsV.GetPrecisions();

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

            zero.ObservedValue = Vector.Zero(numBases);

            a.ObservedValue = 0.5;
            b.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(numBases, 1e-6);

            // TODO: Try the two different priors for beta here
            beta.ObservedValue = PositiveDefiniteMatrix.Identity(numBases);
            //beta.ObservedValue = PositiveDefiniteMatrix.IdentityScaledBy(numBases, 1e-8);
            //Console.WriteLine($"Using sigma={sigma:N2}, bound={bound:N2}, beta={beta.ObservedValue:N2}");
        }
    }
}

