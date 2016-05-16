//
// Results.cs
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
    using System.Collections.Generic;

    /// <summary>
    /// Results of experiments with different numbers of bases
    /// </summary>
    public class Results : Serializable<Results>
    {
        public bool Normalised { get; set; }

        /// <summary>
        /// Gets or sets the basis counts
        /// </summary>
        public IList<int> BasisCounts { get; set; } = new List<int>();

        /// <summary>
        /// Gets or sets the errors.
        /// </summary>
        public IList<double> Errors { get; set; } = new List<double>();

        /// <summary>
        /// Gets or sets the sparsity level (on the training data).
        /// </summary>
        public IList<double> Sparsity { get; set; } = new List<double>();

        /// <summary>
        /// Gets or sets the evidence.
        /// </summary>
        public IList<double> Evidence { get; set; } = new List<double>();

        /// <summary>
        /// Create the results object
        /// </summary>
        public Results()
        {
        }

        /// <summary>
        /// Create the results object
        /// </summary>
        /// <param name="basisCounts">The basis counts.</param>
        public Results(int[] basisCounts)
        {
            if (basisCounts == null || basisCounts.Length == 0)
            {
                throw new ArgumentOutOfRangeException(nameof(basisCounts));
            }

            BasisCounts = basisCounts;
            Errors = new double[basisCounts.Length];
            Evidence = new double[basisCounts.Length];
            Sparsity = new double[basisCounts.Length];
        }
    }
}

