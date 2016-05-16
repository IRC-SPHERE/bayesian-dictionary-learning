//
// BDLParameters.cs
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

    public class BDLParameters : Serializable<BDLParameters>
    {
        public Mode Mode { get; set; }
        public bool NonNegative { get; set; }
        public bool ShowFactorGraph { get; set; }
        public bool Debug { get; set; }
        public bool Sparse { get; set; } = true;
        public bool NormConstraints { get; set; }
        public bool IncludeBias { get; set; }
        public bool MissingData { get; set; }
        public double Tolerance { get; set; } = 1e-3;
        public bool ShowProgress { get; set; } = true;
        public bool UseMatrixMultiply { get; set; } = true;

        public Func<InferenceMonitor, double, bool> ConvergenceCriterion { get; set; } =
            (monitor, tolerance) => 1 - monitor.EvidenceRatio < tolerance;


//        public Dictionary<Mode, int> MinIterations { get; set; } = new Dictionary<Mode, int>
//        {
//            [Mode.Train] = 5,
//            //            [Mode.TrainOnline] = 5,
//            [Mode.TrainFixed] = 3,
//            [Mode.Reconstruct] = 1,
//        };

        public Dictionary<Mode, int> MaxIterations { get; set; } = new Dictionary<Mode, int>
        {
            [Mode.Train] = 100,
//            [Mode.TrainOnline] = 5,
            [Mode.TrainFixed] = 100,
            [Mode.Reconstruct] = 100,
        };
    }
}