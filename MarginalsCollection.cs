//
// MarginalsCollection.cs
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

using System.Linq;

namespace BayesianDictionaryLearning
{
    using InferHelpers;
    using MicrosoftResearch.Infer.Distributions;

    public class MarginalsCollection : Serializable<MarginalsCollection>
    {
        private Marginals trainPosteriors;

        public Marginals TrainPosteriors
        {
            get { return trainPosteriors; }
            set
            {
                trainPosteriors = value;
                if (trainPosteriors == null) return;
                if (trainPosteriors.Coefficients != null)
                {
                    TrainSparsity = trainPosteriors.Coefficients.GetSparsity(1e-2);
                    AverageTrainSparsity = TrainSparsity.Average();
                    AverageCoefficientNorm = TrainPosteriors.Coefficients.Average(ia => ia.L2Norm());
                }

                if (trainPosteriors.Dictionary != null)
                {
                    AverageDictionaryNorm = TrainPosteriors.Dictionary.Average(ia => ia.L2Norm());
                }
            }
        }

        public Marginals TestPosteriors { get; set; }

        public Gaussian[][] Reconstructed { get; set; }

        public double[] TrainSparsity { get; set; }
        public double AverageTrainSparsity { get; set; }

        public double AverageDictionaryNorm { get; set; }
        public double AverageCoefficientNorm { get; set; }
    }
}