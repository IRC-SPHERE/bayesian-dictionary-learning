//
// Helpers.cs
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
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Models;
    using MicrosoftResearch.Infer.Distributions;

    public static class Helpers
    {
        /// <summary>
        /// Constrain norms.
        /// </summary>
        /// <param name="array">The array to constrain.</param>
        /// <param name="sample">Sample range.</param>
        /// <param name="basis">Basis range.</param>
        public static void ConstrainNorms(VariableArray<VariableArray<double>, double[][]> array, Range basis, Range sample)
        {
            using (Variable.ForEach(basis))
            {
                var squares = Variable.Array<double>(sample).Named("squares");
                using (Variable.ForEach(sample))
                {
                    var d = Variable.Copy(array[basis][sample]).Named("d");
                    squares[sample] = array[basis][sample]*d;
                }

                var norm = Variable.Sum(squares).Named("norm");
                Variable.ConstrainEqualRandom(norm, new Gaussian(1.0, 1.0));
            }
        }

        /// <summary>
        /// Matrix multiply factor.
        /// </summary>
        /// <returns>The multiply.</returns>
        /// <param name="coefficients">Coefficients.</param>
        /// <param name="dictionary">Dictionary.</param>
        /// <param name="signal">Signal range.</param>
        /// <param name="sample">Sample range.</param>
        /// <param name="basis">Basis range.</param>
        public static VariableArray2D<double> MatrixMultiply(VariableArray2D<double> coefficients,
            VariableArray2D<double> dictionary,
            Range signal, Range sample, Range basis)
        {
            var cleanSignals = Variable.Array<double>(signal, sample).Named("clean");
            using (Variable.ForEach(signal))
            {
                using (Variable.ForEach(sample))
                {
                    var products = Variable.Array<double>(basis).Named("products");
                    using (Variable.ForEach(basis))
                    {
                        products[basis] = dictionary[basis, sample] * coefficients[signal, basis];
                    }

                    cleanSignals[signal, sample] = Variable.Sum(products);
                }
            }

            return cleanSignals;
        }

        /// <summary>
        /// Matrix multiply factor with missing values.
        /// </summary>
        /// <returns>The multiply.</returns>
        /// <param name="coefficients">Coefficients.</param>
        /// <param name="dictionary">Dictionary.</param>
        /// <param name="signal">Signal range.</param>
        /// <param name="sample">Sample range.</param>
        /// <param name="basis">Basis range.</param>
        public static VariableArray2D<double> MatrixMultiply(VariableArray2D<double> coefficients,
            VariableArray2D<double> dictionary,
            Range signal, Range sample, Range basis, VariableArray2D<bool> missing)
        {
            var cleanSignals = Variable.Array<double>(signal, sample).Named("clean");
            using (Variable.ForEach(signal))
            {
                using (Variable.ForEach(sample))
                {
                    using (Variable.If(missing[signal, sample]))
                    {
                        cleanSignals[signal, sample] = Variable.Observed(0.0);
                    }

                    using (Variable.IfNot(missing[signal, sample]))
                    {
                        var products = Variable.Array<double>(basis).Named("products");
                        using (Variable.ForEach(basis))
                        {
                            products[basis] = dictionary[basis, sample]*coefficients[signal, basis];
                        }

                        cleanSignals[signal, sample] = Variable.Sum(products);
                    }
                }
            }

            return cleanSignals;
        }

        /// <summary>
        /// Matrix multiply factor.
        /// </summary>
        /// <returns>The multiply.</returns>
        /// <param name="coefficients">Coefficients.</param>
        /// <param name="dictionary">Dictionary.</param>
        /// <param name="signal">Signal range.</param>
        /// <param name="sample">Sample range.</param>
        public static VariableArray2D<double> MatrixMultiply(VariableArray<Vector> coefficients,
            VariableArray<Vector> dictionary,
            Range signal, Range sample)
        {
            var cleanSignals = Variable.Array<double>(signal, sample).Named("clean");
            using (Variable.ForEach(signal))
            {
                using (Variable.ForEach(sample))
                {
                    cleanSignals[signal, sample] = Variable.InnerProduct(dictionary[sample], coefficients[signal]);
                }
            }

            return cleanSignals;
        }
    }
}