//
// InferenceProgressHandlers.cs
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
    using System.Linq;
    using Models;
    using InferHelpers;
    using MicrosoftResearch.Infer;
    using PythonPlotter;

    public class InferenceProgressHandlers
    {
        public Experiment RunningExperiment { get; set; }

        public void CustomHandler(object sender, ProgressChangedEventArgs eventArgs)
        {
            if (eventArgs.Iteration == 0)
            {
                return;
            }

            var model = (BDL) RunningExperiment.Models.Train;

            RunningExperiment.ConvergenceResults.Evidence.Add(model.InferenceMonitor.CurrentEvidence);

            string st = $"accel convergence {RunningExperiment.ConvergenceResults.Name} {eventArgs.Iteration}";
            var trainPosteriors = model.GetCurrentMarginals(null);

            // Get the cofficients for the test examples
            var testPosteriors = RunningExperiment.Models.TrainFixed.Train(trainPosteriors,
                RunningExperiment.DataSet.TestSignals.ToRowArrays());

            PlottingHelper.PlotPosteriors(testPosteriors.Dictionary, "test_dictionary", st, 0, 16, 4, 4,
                PlotType.Line, false);
            PlottingHelper.PlotPosteriors(testPosteriors.Coefficients, "test_coefficients", st, 0, 4, 4, 1,
                PlotType.Bar, false);

            // Reconstruct
            var reconstructed = RunningExperiment.Models.Reconstruct.Reconstruct(testPosteriors).Signals;
            if (RunningExperiment.Normalise)
            {
                reconstructed = reconstructed.Normalise();
            }

            var reconstructions = RunningExperiment.DataSet.TestSignals.ToRowArrays().Zip(reconstructed,
                (s, e) => new Reconstruction {Signal = s, Estimate = e}).ToArray();

            // Compute average reconstruction error
            double error = reconstructions.Select(ia => ia.Error).ToArray().Average();
            RunningExperiment.ConvergenceResults.Errors.Add(error);
            Console.WriteLine($"Reconstruction error {error}");

            PlottingHelper.PlotReconstructions(reconstructions, RunningExperiment.ConvergenceResults.Errors.Last(), 2, 2, 1,
                st, false, false);
        }
    }
}