//
// DataSet.cs
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
    using System.Linq;

    using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
    using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;
    using MicrosoftResearch.Infer.Maths;

    public class DataSet
    {
        public string Name { get; set; }

        /// <summary>
        /// The train labels.
        /// </summary>
        private Vector trainLabels;

        /// <summary>
        /// The test labels.
        /// </summary>
        private Vector testLabels;

        /// <summary>
        /// Gets or sets the train signals.
        /// </summary>
        /// <value>The train signals.</value>
        public Matrix TrainSignals { get; set; }

        /// <summary>
        /// Gets or sets the train features.
        /// </summary>
        /// <value>The train features.</value>
        public Matrix TrainFeatures { get; set; }

        /// <summary>
        /// Gets or sets the test signals.
        /// </summary>
        /// <value>The test signals.</value>
        public Matrix TestSignals { get; set; }

        /// <summary>
        /// Gets or sets the test features.
        /// </summary>
        /// <value>The test features.</value>
        public Matrix TestFeatures { get; set; }

        /// <summary>
        /// Gets or sets the train labels.
        /// </summary>
        public Vector TrainLabels
        {
            get 
            {
                return trainLabels;
            }
            set 
            {
                trainLabels = value;

                if (trainLabels == null)
                {
                    TrainLabelsBinary = null;
                    TrainLabelsInteger = null;
                    TrainClasses = null;
                    TrainLabelCounts = null;
                }
                else
                {
                    var trainGroups = trainLabels.GroupBy(x => (int)x).OrderBy(ia => ia.Key);
                    TrainClasses = trainGroups.Select(ia => ia.Key).ToArray();
                    TrainLabelCounts = trainGroups.Select(ia => $"[{ia.Key}: {ia.Count()}]");

                    TrainLabelsBinary = new bool[trainLabels.Count];
                    TrainLabelsInteger = new int[trainLabels.Count];
                    for (int i = 0; i < trainLabels.Count; i++)
                    {
                        TrainLabelsBinary[i] = Math.Abs(trainLabels[i] - PositiveClass) < double.Epsilon;
                        TrainLabelsInteger[i] = Array.FindIndex(TrainClasses, x => x == (int)trainLabels[i]);
                    }
                }
            }
        }

        /// <summary>
        /// Gets the test labels.
        /// </summary>
        public Vector TestLabels
        {
            get 
            { 
                return testLabels; 
            }
            set 
            { 
                testLabels = value; 

                if (testLabels == null)
                {
                    TestLabelsBinary = null;
                    TestLabelsInteger = null;
                    TestClasses = null;
                    TestLabelCounts = null;
                }
                else
                {
                    var testGroups = testLabels.GroupBy(x => (int)x).OrderBy(ia => ia.Key);
                    TestClasses = testGroups.Select(ia => ia.Key).ToArray();
                    TestLabelCounts = testGroups.Select(ia => $"[{ia.Key}: {ia.Count()}]");

                    TestLabelsBinary = new bool[testLabels.Count];
                    TestLabelsInteger = new int[testLabels.Count];
                    for (int i = 0; i < testLabels.Count; i++)
                    {
                        TestLabelsBinary[i] = Math.Abs(testLabels[i] - PositiveClass) < double.Epsilon;
                        TestLabelsInteger[i] = Array.FindIndex(TestClasses, x => x == (int)testLabels[i]);
                    }
                }
            }
        }

        /// <summary>
        /// Gets or sets the positive class (binary classification only).
        /// </summary>
        public int PositiveClass { get; set; }
        
        /// <summary>
        /// Gets the train labels in binary format. 
        /// </summary>
        public bool[] TrainLabelsBinary { get; set; }
        
        /// <summary>
        /// Gets the test labels in binary format. 
        /// </summary>
        public bool[] TestLabelsBinary { get; set; }

        /// <summary>
        /// Gets the train labels in ingeger format. 
        /// </summary>
        public int[] TrainLabelsInteger { get; set; }

        /// <summary>
        /// Gets the test labels in integer format. 
        /// </summary>
        public int[] TestLabelsInteger { get; set; }

        /// <summary>
        /// Gets the train classes.
        /// </summary>
        /// <value>The train classes.</value>
        public int[] TrainClasses { get; set; }

        /// <summary>
        /// Gets the test classes.
        /// </summary>
        /// <value>The test classes.</value>
        public int[] TestClasses { get; set; }

        /// <summary>
        /// Gets the train label counts.
        /// </summary>
        /// <value>The train label counts.</value>
        public IEnumerable<string> TrainLabelCounts { get; set; }

        /// <summary>
        /// Gets the test label counts.
        /// </summary>
        /// <value>The test label counts.</value>
        public IEnumerable<string> TestLabelCounts { get; set; }

        /// <summary>
        /// Normalise this instance.
        /// </summary>
        public void Normalise()
        {
            TrainSignals = TrainSignals.NormalizeRows(2.0);
            TestSignals = TestSignals.NormalizeRows(2.0);
        }

        /// <summary>
        /// Gets the random subsets.
        /// </summary>
        /// <returns>The random subsets.</returns>
        /// <param name="numTrain">Number train.</param>
        /// <param name="numTest">Number test.</param>
        /// <param name="testSignals">Test signals.</param>
        /// <param name="trainSignals">Train signals.</param>
        /// <param name="trainFeatures">Train features.</param>
        /// <param name="trainLabels">Train labels.</param>
        /// <param name="testFeatures">Test features.</param>
        /// <param name="testLabels">Test labels.</param>
        public void GetRandomSubsets(int numTrain, int numTest, 
                                     out double[][] trainSignals, out double[][] trainFeatures, out int[] trainLabels, 
                                     out double[][] testSignals, out double[][] testFeatures, out int[] testLabels)
        {
            var trainIndices = Rand.Perm(Math.Min(numTrain, TrainSignals.RowCount));
            trainSignals = trainIndices.Select(i => TrainSignals.Row(i).ToArray()).ToArray();
            trainFeatures = TrainFeatures == null ? null : trainIndices.Select(i => TrainFeatures.Row(i).ToArray()).ToArray();
            trainLabels = trainIndices.Select(i => TrainLabelsInteger[i]).ToArray();

            var testIndices = Rand.Perm(Math.Min(numTest, TestSignals.RowCount));
            testSignals = testIndices.Select(i => TestSignals.Row(i).ToArray()).ToArray();
            testFeatures = TestFeatures == null ? null : testIndices.Select(i => TestFeatures.Row(i).ToArray()).ToArray();
            testLabels = testIndices.Select(i => TestLabelsInteger[i]).ToArray();
        }

        /// <summary>
        /// Prints the summary.
        /// </summary>
        /// <returns>The summary.</returns>
        public void PrintSummary()
        {
            Console.WriteLine($"# train signals: {TrainSignals.RowCount}");
            Console.WriteLine($"# test signals: {TestSignals.RowCount}");

            Console.WriteLine($"signal width: {TrainSignals.ColumnCount}");
            if (TrainFeatures != null)
            {
                Console.WriteLine($"# features: {TrainFeatures.ColumnCount}");
            }

			if (TrainLabels != null)
			{
				Console.WriteLine($"Train Label Counts: {string.Join(", ", TrainLabelCounts)}");
			}

			if (TestLabels != null)
			{
            	Console.WriteLine($"Test Label Counts: {string.Join(", ", TestLabelCounts)}");
			}
        }
    }    
}