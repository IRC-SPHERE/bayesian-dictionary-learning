//
// Data.cs
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
    using System.IO;
    using System.Linq;
    using MathNet.Numerics;
    using MathNet.Numerics.Data.Text;
    using MathNet.Numerics.Data.Matlab;
    using MicrosoftResearch.Infer.Maths;
    
    using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
    using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

    /// <summary>
    /// Data loading and generation.
    /// </summary>
    public static class Data
    {
        /// <summary>
        /// The data path.
        /// </summary>
        public const string DataPath = "../../data";

        /// <summary>
        /// The name of the data file.
        /// </summary>
        public const string DataFileName = "data.mat";

        /// <summary>
        /// Loads the train test data.
        /// </summary>
        /// <returns>The train test data.</returns>
        /// <param name="dataSetName">The name of the data set.</param>
        /// <param name="trainLimit"></param>
        /// <param name="testLimit"></param>
        /// <param name="plot"></param>
        /// <param name="permute">Whether to perumute the instances.</param>
        public static DataSet LoadTrainTestData(string dataSetName, int trainLimit = int.MaxValue,
            int testLimit = int.MaxValue, bool plot = true, bool permute = false)
        {
            string path = Path.Combine(DataPath, dataSetName, DataFileName);
            Console.WriteLine($"Loading {path}");
            var ms = MatlabReader.ReadAll<double>(path, "train_x", "train_y", "test_x", "test_y");

            DataSet dataSet;
            // TODO: Need to perumute data and labels in the same way!
            if (permute)
            {
                Func<Random> rng = () => new Random(0);
                dataSet = new DataSet
                {
                    Name = dataSetName,
                    TrainSignals = Matrix.Build.DenseOfRows(ms["train_x"].EnumerateRows().SelectPermutation(rng()).Take(trainLimit)),
                    TrainLabels = Vector.Build.DenseOfEnumerable(ms["train_y"].Row(0).SelectPermutation(rng()).Take(trainLimit)),
                    TestSignals = Matrix.Build.DenseOfRows(ms["test_x"].EnumerateRows().SelectPermutation(rng()).Take(testLimit)),
                    TestLabels = Vector.Build.DenseOfEnumerable(ms["test_y"].Row(0).SelectPermutation(rng()).Take(testLimit))
                };
            }
            else
            {
                dataSet = new DataSet
                {
                    Name = dataSetName,
                    TrainSignals = Matrix.Build.DenseOfRows(ms["train_x"].EnumerateRows().Take(trainLimit)),
                    TrainLabels = Vector.Build.DenseOfEnumerable(ms["train_y"].Row(0).Take(trainLimit)),
                    TestSignals = Matrix.Build.DenseOfRows(ms["test_x"].EnumerateRows().Take(testLimit)),
                    TestLabels = Vector.Build.DenseOfEnumerable(ms["test_y"].Row(0).Take(testLimit))
                };
            }

            if (plot)
            {
                PlottingHelper.PlotImages(dataSet.TrainSignals.SubMatrix(0, 16, 0, dataSet.TrainSignals.ColumnCount),
                    "dataSetName", 4, 4, true);
            }

            Console.WriteLine("Done.");
            return dataSet;
        }

        //      /// <summary>
        //      /// Loads the precomputed dictionaries/coefficients.
        //      /// </summary>
        //      /// <returns>The precomputed dictionaries/coefficients.</returns>
        //      /// <param name="path">Path.</param>
        //      public static IEnumerable<Precomputed> LoadPrecomputed(string path, int limit = int.MaxValue, bool normalise = true)
        //{
        //	const int basisMin = 2;
        //	const int basisMax = 12;
        //	const int regMin = 0;
        //	const int regMax = 5;
        //	const bool pos = false;
        //	const bool lasso = true;
        //	const bool cc = false;

        //          var bases = Enumerable.Range(basisMin, basisMax - basisMin).Select(x => (int)Math.Pow(2.0, x)).ToArray();
        //	var regs = Enumerable.Range(regMin, regMax - regMin).Select(x => Math.Pow(10.0, -x)).ToArray();

        //          foreach (var k in bases)
        //	{
        //		foreach (var lambda in regs)
        //		{
        //                  Precomputed pd;
        //                  try
        //                  {
        //                      pd = Precomputed.Load(path, k, lambda, cc, pos, lasso, limit, normalise);
        //                  }
        //                  catch (IOException e)
        //                  {
        //                      Console.WriteLine(e.Message);
        //                      continue;
        //                  }

        //                  yield return pd;
        //		}
        //	}
        //}


        public static Vector Signal1(int signalWidth, bool nrm, double noise)
        {
            var v = Vector.Build.DenseOfEnumerable(Generate.Map2(
                Generate.LinearSpacedMap(signalWidth, -10.0, 10.0, Math.Sin),
                Generate.LinearSpacedMap(signalWidth, -50.0, 50.0, Math.Sin),
                (a, b) => a + b + (Rand.Normal()*noise)));
            return nrm ? v/v.L2Norm() : v;
        }

        public static Vector Signal2(int signalWidth, double slow, double fast, bool nrm, double noise)
        {
            var v = Vector.Build.DenseOfEnumerable(Generate.Map(
                Enumerable.Concat(
                    Generate.LinearSpacedMap(signalWidth/2, -Math.PI*slow, Math.PI*slow, Math.Sin),
                    Generate.LinearSpacedMap(signalWidth/2, -Math.PI*fast, Math.PI*fast, Math.Sin)).ToArray(),
                a => a + (Rand.Normal()*noise)));
            return nrm ? v/v.L2Norm() : v;
        }

        /// <summary>
        /// Generates the signals.
        /// </summary>
        /// <returns>The signals.</returns>
        /// <param name="basisWidth">Basis width.</param>
        /// <param name="copies"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="normalise"></param>
        public static double[][] GenerateSignals(int basisWidth, int copies, double noiseVariance, bool normalise = true)
        {
            Rand.Restart(1234);

            const double slow = 4;
            const double fast = 10;
            const int signalWidth = 128;

            // Generate a couple of interesting signals
            var signalList = new List<double[]>();
            if (basisWidth < signalWidth)
            {
                var x1X = Signal1(signalWidth, normalise, noiseVariance);
                var x2X = Signal2(signalWidth, slow, fast, normalise, noiseVariance);

                // Sliding windows
                for (var i = 0; i < signalWidth - basisWidth; i++)
                {
                    signalList.Add(x1X.SubVector(i, basisWidth).ToArray());
                    signalList.Add(x2X.SubVector(i, basisWidth).ToArray());
                }
            }
            else
            {
                for (int i = 0; i < copies; i++)
                {
                    signalList.Add(Signal1(signalWidth, normalise, noiseVariance).ToArray());
                    signalList.Add(Signal2(signalWidth, slow, fast, normalise, noiseVariance).ToArray());
                }
            }
            
            return signalList.ToArray(); 
        }



        /// <summary>
        /// Gets the toy data.
        /// </summary>
        /// <returns>The toy data.</returns>
        public static DataSet GetToyData(int basisWidth, int count, double noise)
        {
            var signals = Matrix.Build.DenseOfRowArrays(GenerateSignals(basisWidth, count, noise));
            return new DataSet
            {
                Name = "toy",
                TrainSignals = signals,
                TestSignals = signals
            };
        }
        
        /// <summary>
        /// Load Accelerometer Data
        /// </summary>
        /// <returns>Signals as a jagged array.</returns>
        public static Matrix LoadAccelerometerData(int maxRows, bool normalise)
        {
            var filename = Path.Combine(DataPath, "accel", "mag.csv");
            var matrix = DelimitedReader.Read<double>(filename, false, ",", false);
            if (normalise)
            {
                matrix = matrix.NormalizeRows(2.0);
            }

            Console.WriteLine($"Loaded data, shape={matrix}");
            
            return maxRows > 0 
                ? matrix.SubMatrix(0, Math.Min(maxRows, matrix.RowCount), 0, matrix.ColumnCount)
                : matrix;
        }
        
        /// <summary>
        /// Load Accelerometer Data
        /// </summary>
        /// <returns>Signals as a jagged array.</returns>
        public static void LoadAccelerometerData(int maxRows, bool normalise,
            out Matrix signals, out Vector subjects, out Vector labels)
        {
            string filename = Path.Combine(DataPath, "accel", "mag0.csv");
            var matrix = DelimitedReader.Read<double>(filename, false, ",", false);
            
            signals = matrix.SubMatrix(0, Math.Min(maxRows, matrix.RowCount), 2, matrix.ColumnCount - 2);
            
            if (normalise)
            {
                signals = signals.NormalizeRows(2.0);
            }
            
            subjects = matrix.Column(0).SubVector(0, Math.Min(maxRows, matrix.RowCount));
            labels = matrix.Column(1).SubVector(0, Math.Min(maxRows, matrix.RowCount));

            Console.WriteLine($"Loaded data, shape={matrix}");
        }

        /// <summary>
        /// Load Accelerometer Data in separate train and test sets. Splits by subject
        /// </summary>
        /// <returns>Signals as a jagged array.</returns>
        public static IEnumerable<DataSet> LoadAccelerometerData(int train, int test, bool normalise)
        {
            string filename = Path.Combine(DataPath, "accel", "mag0.csv");

            // Each row of data is of form [subject, label, x_0, ..., x_n] 
            // Class labels: walking/upstairs/downstairs/standing/sitting/lying
            
            var matrix = DelimitedReader.Read<double>(filename, false, ",", hasHeaders: false);
            Console.WriteLine($"Loaded data, shape={matrix}");
            
            if (train + test > matrix.RowCount)
            {
                throw new ArgumentOutOfRangeException(nameof(train));
            }
            
            // Get list of subjects
            var subjects = matrix.Column(0).Select(ia => (int)ia).Distinct().ToArray();
            
            // Choose subjects 0 and 1 and labels 0 and 1
            // var subjectsToUse = new[] { 0, 2 };
            var labelsToUse = new[] { 0, 5 };
            // return GetSubSet(matrix, subjectsToUser, labelsToUse, normalise, train, test);
            return
                from subject in subjects
                select new[] { subject } into testSubjects
                let trainSubjects = subjects.Except(testSubjects).ToArray()
                select GetSubSet(matrix, trainSubjects, testSubjects, labelsToUse, normalise, train, test);
        }

        /// <summary>
        /// Gets the sub set.
        /// </summary>
        /// <returns>The sub set.</returns>
        /// <param name="matrix">Matrix.</param>
        /// <param name="trainSubjects">Train subjects.</param>
        /// <param name="testSubjects">Test subjects.</param>
        /// <param name="labelsToUse">Labels to use.</param>
        /// <param name="normalise">Normalise.</param>
        /// <param name="train">Train.</param>
        /// <param name="test">Test.</param>
        public static DataSet GetSubSet(Matrix matrix, int[] trainSubjects, int[] testSubjects, int[] labelsToUse, bool normalise,
            int train = int.MaxValue, int test = int.MaxValue)
        {
            var trainRows = Matrix.Build.DenseOfRowVectors(
                matrix.EnumerateRowsIndexed()
                    .OrderBy(ia => Rand.Double())
                    .Where(ia => labelsToUse.Any(x => x == (int)ia.Item2[1]) && trainSubjects.Any(x => x == (int)ia.Item2[0]))
                    .Take(train)
                    .Select(ia => ia.Item2).ToArray());
            
            var testRows = Matrix.Build.DenseOfRowVectors(
                matrix.EnumerateRowsIndexed()
                    .OrderBy(ia => Rand.Double())
                    .Where(ia => labelsToUse.Any(x => x == (int)ia.Item2[1]) && testSubjects.Any(x => x == (int)ia.Item2[0]))
                    .Take(train)
                    .Select(ia => ia.Item2).ToArray());
            
            var dataSet = new DataSet 
            {
                TrainSignals = trainRows.SubMatrix(0, trainRows.RowCount, 2, trainRows.ColumnCount - 2),
                TestSignals = testRows.SubMatrix(0, testRows.RowCount, 2, testRows.ColumnCount - 2),
                TrainLabels = trainRows.Column(1),
                TestLabels = testRows.Column(1)
            };
            
            if (normalise)
            {
                dataSet.Normalise();
            }
            
            return dataSet;
        }

        /// <summary>
        /// Load Accelerometer Data
        /// </summary>
        /// <param name="maxRows">Maximum rows to retrieve.</param>
        /// <param name="normalise">Whether to normalise the rows.</param>
        /// <param name="signalWidth">The signal width.</param>
        /// <param name="signals">The data</param>
        /// <param name="subjects"></param>
        /// <param name="labels"></param>
        public static void LoadAccelerometerData(int maxRows, bool normalise, int signalWidth,
            out Matrix signals, out Vector subjects, out Vector labels)
        {
            string filename = Path.Combine(DataPath, "accel", "data", "xyz_50.csv");
            // Format as follows:
            // ,label,repetition,subject,x,y,z
            // 0,0,1,0,1.3878650000000008,-0.3367674000000004,-0.057304429999999996
            var matrix = DelimitedReader.Read<double>(filename, sparse: false, delimiter: ",", hasHeaders: true);
            // var xyz = matrix.SubMatrix(0, Math.Min(maxRows, matrix.RowCount), 4, 3);
            var mag = matrix.RowNorms(2.0);
            var mag0 = mag - 1;
            
            // Now reshape into chunks the size of the signal width
            int numSignals = Math.Min(maxRows, mag0.Count / signalWidth);
            var xx = new Vector[numSignals];
            for (var i = 0; i < numSignals; i++)
            {
                xx[i] = mag0.SubVector(i * signalWidth, signalWidth);
            }
            
            signals = Matrix.Build.DenseOfRowVectors(xx);
            
            if (normalise)
            {
                signals.NormalizeRows(2.0);
            }
            
            subjects = matrix.Column(3);
            labels = matrix.Column(1);
        }
        
        /// <summary>
        /// Load Precomputed Dictionary
        /// </summary>
        /// <param name="bases">Bases.</param>
        /// <param name="lambda">The regularisation parameter.</param>
        /// <param name="normalise">Normalise.</param>
        public static Matrix LoadDictionary(int bases, double lambda, bool normalise)
        {
            string filename = Path.Combine(DataPath, "accel", $"dictionary_k={bases}_lambda={lambda}.csv");
            var matrix = DelimitedReader.Read<double>(filename, false, ",", false);
            if (normalise)
            {
                matrix = matrix.NormalizeRows(2.0);
            }

             Console.WriteLine($"Loaded dictionary, shape={matrix}");
            
            return matrix;
        }

        /// <summary>
        /// Loads the coefficients.
        /// </summary>
        /// <returns>The coefficients.</returns>
        /// <param name="bases">Bases.</param>
        /// <param name="lambda">The regularisation parameter.</param>
        /// <param name="normalise">Normalise.</param>
        public static Matrix LoadCoefficients(int bases, double lambda, bool normalise = false)
        {
            string filename = Path.Combine(DataPath, "accel", $"coefficients_k={bases}_lambda={lambda}.csv");
            var matrix = DelimitedReader.Read<double>(filename, false, ",");
            if (normalise)
            {
                matrix = matrix.NormalizeRows(2.0);
            }

            Console.WriteLine($"Loaded coefficients, shape={matrix}");
            
            return matrix;
        }

		/// <summary>
		/// Reshape the specified flat image.
		/// </summary>
		/// <param name="imageFlat">Image flat.</param>
		public static double[][] Reshape(Vector imageFlat)
		{
			var sz = (int)Math.Sqrt(imageFlat.Count);
			var image = new double[sz][];
			for (int i = 0; i < sz; i++)
			{
				image[i] = imageFlat.SubVector(i * sz, sz).ToArray();
			}

			return image;
		}
    }
}