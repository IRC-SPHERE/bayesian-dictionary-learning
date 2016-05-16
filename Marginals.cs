//
// Marginals.cs
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
    using InferHelpers;
    using MicrosoftResearch.Infer.Distributions;
    using static MathNet.Numerics.LinearAlgebra.Vector<double>;
    using MathNet.Numerics.LinearAlgebra;
    using MicrosoftResearch.Infer.Maths;
    
    /// <summary>
    /// Marginals
    /// </summary>
    [Serializable]
    public class Marginals
    {
        /// <summary>
        /// The sparse coding coeffficients
        /// </summary>
        public Gaussian[][] Coefficients { get; set; }

        /// Gets or sets the noise precision.
        /// </summary>
        /// <value>The noise precision.</value>
        public Gamma NoisePrecision { get; set; }

        /// <summary>
        /// The dictionary marginals
        /// </summary>
        public Gaussian[][] Dictionary { get; set; }

        /// <summary>
        /// The dictionary means marginals
        /// </summary>
        public Gaussian[][] DictionaryMeans { get; set; }

        /// <summary>
        /// The dictionary precisions marginals
        /// </summary>
        public Gamma[][] DictionaryPrecisions { get; set; }

        /// <summary>
        /// The model evidence
        /// </summary>
        public Bernoulli Evidence { get; set; }

        /// <summary>
        /// Marginal distribution over signals (for reconstruction)
        /// </summary>
        public Gaussian[][] Signals { get; set; }

        /// <summary>
        /// Prints the summary.
        /// </summary>
        public void PrintSummary()
        {
            Console.WriteLine($"Reconstruct Noise precision: {NoisePrecision}");

            // Print norms of dictionary elements
            for (var i = 0; i < Dictionary.Length; i++)
            {
                var basis = Build.DenseOfEnumerable(Dictionary[i].GetMeans());
                Console.WriteLine($"Basis {i}, L2 {basis.L2Norm():N2}, L∞ {basis.InfinityNorm():N2}");
            }
        }

        /// <summary>
        /// Creates the priors.
        /// </summary>
        /// <returns>The priors.</returns>
        /// <param name="dictionary">Precomputed dictionary.</param>
        /// <param name="coefficients">Precompute coefficients.</param>
        /// <param name="basisWidth">Basis width.</param>
        /// <param name="numBases">Number bases.</param>
        /// <param name="sigma">Sigma.</param>
        /// <param name="vectorized">Indicates whether this is the vectorized form (full covariance).</param>
        public static Marginals CreatePriors(Matrix<double> dictionary, Matrix<double> coefficients, int basisWidth,
            int numBases, double sigma,
            bool vectorized)
        {
            return new Marginals
                {
                    Dictionary = dictionary == null
                        // ? DistributionHelpers.CreateGaussianArray(numBases, basisWidth, 0.0, sigma * sigma)
                        ? DistributionHelpers.CreateGaussianArray(numBases, basisWidth, Rand.Normal, sigma*sigma)
                        : DistributionHelpers.GetGaussianArray(dictionary.ToRowArrays()),
                    Coefficients = coefficients == null
                        ? null
                        : DistributionHelpers.GetGaussianArray(coefficients.ToRowArrays())
                };
        }

        /// <summary>
        /// Creates the hyper-priors.
        /// </summary>
        /// <returns>The priors.</returns>
        /// <param name="dictionary">Precomputed dictionary.</param>
        /// <param name="coefficients">Precompute coefficients.</param>
        /// <param name="basisWidth">Basis width.</param>
        /// <param name="numBases">Number bases.</param>
        public static Marginals CreateHyperPriors(Matrix<double> dictionary, Matrix<double> coefficients, int basisWidth, int numBases)
        {
            var priors = CreatePriors(dictionary, coefficients, basisWidth, numBases, 1.0, false);
            priors.DictionaryMeans = DistributionHelpers.CreateGaussianArray(numBases, basisWidth, 0.0, 1.0);
            priors.DictionaryPrecisions = DistributionHelpers.CreateGammaArray(numBases, basisWidth, 1.0, 1.0);
            return priors;
        }
    }
}
