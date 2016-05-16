//
// Precomputed.cs
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

using System;

namespace BayesianDictionaryLearning
{
    using System.IO;
    using System.Linq;
    using MathNet.Numerics.Data.Matlab;
    using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

    /// <summary>
    /// Precomputed.
    /// </summary>
    public class Precomputed
    {
        /// <summary>
        /// Gets or sets the number of bases.
        /// </summary>
        /// <value>The number of bases.</value>
        public int NumberOfBases { get; set; }

        /// <summary>
        /// Gets or sets the lambda.
        /// </summary>
        /// <value>The lambda.</value>
        public double Lambda { get; set; }

        /// <summary>
        /// Gets or sets the class conditional.
        /// </summary>
        /// <value>The class conditional.</value>
        public bool ClassConditional { get; set; } = false;

        /// <summary>
        /// Gets or sets the positivity constraints.
        /// </summary>
        /// <value>The positivity constraints.</value>
        public bool PositivityConstraints { get; set; } = false;

        /// <summary>
        /// Gets or sets the lasso.
        /// </summary>
        /// <value>The lasso.</value>
        public bool Lasso { get; set; } = true;

        /// <summary>
        /// Gets or sets the dictionary.
        /// </summary>
        /// <value>The dictionary.</value>
        public Matrix Dictionary { get; set; }

        /// <summary>
        /// Gets or sets the train coefficients.
        /// </summary>
        /// <value>The train coefficients.</value>
        public Matrix TrainCoefficients { get; set; }

        /// <summary>
        /// Gets or sets the test coefficients.
        /// </summary>
        /// <value>The test coefficients.</value>
        public Matrix TestCoefficients { get; set; }

        /// <summary>
        /// Gets the key.
        /// </summary>
        /// <value>The key.</value>
        public string Key
        {
            get 
            {
                string lambda = Lambda >= 0.1 ? $"{Lambda:N1}" : $"{Lambda}";
                return
                    $"k={NumberOfBases}_lambda={lambda}_cc={ClassConditional}_pos={PositivityConstraints}_lasso={Lasso}";
            }
        }

        /// <summary>
        /// Load the specified path, k, lambda, cc, pos, lasso and normalise.
        /// </summary>
        /// <param name="dataSetName">The name of the data set.</param>
        /// <param name="k">The number of bases.</param>
        /// <param name="lambda">Lambda.</param>
        /// <param name="cc">Cc.</param>
        /// <param name="pos">Position.</param>
        /// <param name="lasso">Lasso.</param>
        /// <param name="trainLimit">The limit on the number of examples to load (train coefficients).</param>
        /// <param name="testLimit">The limit on the number of examples to load (test coefficients).</param>
        /// <param name="normalise">Normalise.</param>
        public static Precomputed Load(string dataSetName, int k, double lambda, bool cc = false, bool pos = false,
            bool lasso = true, int trainLimit = int.MaxValue, int testLimit = int.MaxValue, bool normalise = true)
        {
            string path = Path.Combine(Data.DataPath, dataSetName);

            var data = new Precomputed
            {
                NumberOfBases = k,
                Lambda = lambda,
                ClassConditional = cc,
                PositivityConstraints = pos,
                Lasso = lasso
            };

            // Some helper functions
            Func<string, Matrix> load1 = f => MatlabReader.Read<double>(Path.Combine(path, f));
            Func<string, string, Matrix> load2 = (f, part) => MatlabReader.Read<double>(Path.Combine(path, f), part);
            Func<Matrix, int, Matrix> take = (m, rows) => Matrix.Build.DenseOfRows(m.EnumerateRows().Take(rows));

            data.Dictionary = load1($"dictionary_{data.Key}.mat");
            data.TrainCoefficients = take(load2($"coefficients_{data.Key}.mat", "train_z"), trainLimit);
            data.TestCoefficients = take(load2($"coefficients_{data.Key}.mat", "test_z"), testLimit);

            if (normalise)
            {
                data.Dictionary = data.Dictionary.NormalizeRows(2);
                // TODO: Normalise coefficients?
            }

            return data;
        }
    }
}

