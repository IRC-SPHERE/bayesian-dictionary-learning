//
// ClassifierMapping.cs
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
    using System.Collections.Generic;
    using MicrosoftResearch.Infer.Maths;
    using MicrosoftResearch.Infer.Learners.Mappings;

    /// <summary>
    /// A mapping for the Bayes Point Machine classifier tutorial.
    /// </summary>
    public class ClassifierMapping
        : IClassifierMapping<IList<Vector>, int, IList<string>, string, Vector>
    {
        public IEnumerable<int> GetInstances(IList<Vector> featureVectors)
        {
            for (int instance = 0; instance < featureVectors.Count; instance++)
            {
                yield return instance;
            }
        }

        public Vector GetFeatures(int instance, IList<Vector> featureVectors)
        {
            return featureVectors[instance];
        }

        public string GetLabel(
            int instance, IList<Vector> featureVectors, IList<string> labels)
        {
            return labels[instance];
        }

        public IEnumerable<string> GetClassLabels(
            IList<Vector> featureVectors = null, IList<string> labels = null)
        {
            return new[]
            {
                "Walking",
                "Ascending stairs",
                "Descending stairs",
                "Sitting",
                "Standing",
                "Lying down"
            };
        }
    }
}