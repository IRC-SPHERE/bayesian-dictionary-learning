//
// PlottingHelpers.cs
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
    using MicrosoftResearch.Infer.Distributions;
    using PythonPlotter;
    using InferHelpers;

    using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
    using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

    public static class PlottingHelper
    {
        /// <summary>
        /// Plots the results.
        /// </summary>
        /// <param name="x">The x values.</param>
        /// <param name="y">The y values.</param>
        /// <param name="title">The plot title.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="xlabel">x-axis label.</param>
        /// <param name="ylabel">y-axis label.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void Plot(IEnumerable<double> x, IEnumerable<double> y, string title, string subTitle,
            string xlabel, string ylabel, bool show = false)
        {
            var series = (ISeries) (new LineSeries { X = x, Y = y });

            var plotter = new Plotter
            {
                Title = title + (string.IsNullOrEmpty(subTitle) ? string.Empty : " " + subTitle),
                XLabel = xlabel,
                YLabel = ylabel,
                Series = new[] { series },
                ScriptName = Path.Combine(MainClass.ScriptPath, title.Replace(" ", "_") + ".py"),
                FigureName = Path.Combine(MainClass.FigurePath, title.Replace(" ", "_") + ".pdf"),
                Python = MainClass.PythonPath,
                Show = show,
                Tight = true
            };

            plotter.Plot();
        }

        public static void TwinTwinPlot(
            Dictionary<string, IEnumerable<double>> y1,
            Dictionary<string, IEnumerable<double>> y2,
            string title,
            string xlabel,
            string y1Label,
            string y2Label,
            bool show = false)
        {
		    var series1 = y1.Select(ia => (ISeries) (new LineSeries {Label = ia.Key, X = ia.Value})).ToArray();
            var series2 = y2.Select(ia => (ISeries) (new LineSeries {Label = ia.Key, X = ia.Value})).ToArray();

            // Turn on color cycling for both series
			series1[0].Color = "next(palette)";
			series2[0].Color = "next(palette)";

			// Here we build the plotting script for the second plot (without the pre/postamble),
			// so we can append it to the script for the first plot
            var plotter2 = new Plotter { XLabel = xlabel, YLabel = y2Label, Series = series2, TwinX = true };
            plotter2.BuildScript();

            // TODO: http://matplotlib.org/examples/api/two_scales.html

            var plotter1 = new Plotter
            {
                Title = title,
                XLabel = xlabel,
                YLabel = y1Label,
                Series = series1,
                Python = MainClass.PythonPath,
                Show = show,
                Tight = true
            };
            plotter1.Plot(plotter2.Script);
        }

        /// <summary>
        /// Plots the results.
        /// </summary>
        /// <param name="y">The values.</param>
        /// <param name="filename">The file name.</param>
        /// <param name="xlabel">x-axis label.</param>
        /// <param name="ylabel">y-axis label.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void Plot(Dictionary<string, IEnumerable<double>> y, string filename, string xlabel,
            string ylabel, bool show = false)
        {
            var series = y.Select(ia => (ISeries) (new LineSeries {Label = ia.Key, X = ia.Value})).ToArray();

            var plotter = new Plotter
            {
                XLabel = xlabel,
                YLabel = ylabel,
                Series = series,
                ScriptName = Path.Combine(MainClass.ScriptPath, filename.Replace(" ", "_") + ".py"),
                FigureName = Path.Combine(MainClass.FigurePath, filename.Replace(" ", "_") + ".pdf"),
                Python = MainClass.PythonPath,
                Show = show,
                Tight = true
            };

            plotter.Plot();
        }

        /// <summary>
        /// Plots the results.
        /// </summary>
        /// <param name="y">The values.</param>
        /// <param name="title">The plot title.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="xlabel">x-axis label.</param>
        /// <param name="ylabel">y-axis label.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void Plot(Dictionary<string, IEnumerable<double>> y, string title, string subTitle, string xlabel,
            string ylabel, bool show = false)
        {
            var series = y.Select(ia => (ISeries) (new LineSeries {Label = ia.Key, X = ia.Value})).ToArray();

            var plotter = new Plotter
            {
                Title = title + (string.IsNullOrEmpty(subTitle) ? string.Empty : " " + subTitle),
                XLabel = xlabel,
                YLabel = ylabel,
                Series = series,
                ScriptName = Path.Combine(MainClass.ScriptPath, title.Replace(" ", "_") + ".py"),
                FigureName = Path.Combine(MainClass.FigurePath, title.Replace(" ", "_") + ".pdf"),
                Python = MainClass.PythonPath,
                Show = show,
                Tight = true
            };

            plotter.Plot();
        }

        public static void SparsityPlot(Gaussian[][] coefficients, string title, string filename, bool show = false)
        {
            var values = coefficients.GetMeans<Gaussian>(); // .To2D().Transpose().ToJagged();
	        var plotter = new Plotter
				{
					// Title = title,
					XLabel = "bases",
					YLabel = "signals",
                    ScriptName = Path.Combine(MainClass.ScriptPath, $"{filename.Replace(" ", "_")}.py"),
                    FigureName = Path.Combine(MainClass.FigurePath, $"{filename.Replace(" ", "_")}.pdf"),
                    Python = MainClass.PythonPath,
                    Series = new ISeries[] { new HintonSeries { Values = values } },
					Grid = false,
				    Show = show,
				    Tight = true
				};
			plotter.Plot();
        }

        /// <summary>
        /// Plots the functions.
        /// </summary>
        /// <param name="functions">The functions to plot.</param>
        /// <param name="title">The plot title..</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="numToShow">The number of functions to show.</param>
        /// <param name="rows">The number of rows in the subplot.</param>
        /// <param name="cols">The number of columns in the subplot.</param>
        public static void PlotFunctions(double[][] functions, string title, string subTitle, int numToShow, int rows, int cols)
        {
            IList<ISeries> series = functions.Take(numToShow).Select(
                (ia, i) => (ISeries)(new LineSeries
                { 
                    X = ia, 
                    Row = i / cols,
                    Column = i % cols 
                })).ToArray();
            var subplots = new Subplots { Rows = rows, Columns = cols, ShareX = true, ShareY = true };
            var plotter = new Plotter 
            { 
                Title = title + (string.IsNullOrEmpty(subTitle) ? string.Empty : " " + subTitle),
                XLabel = "x", YLabel = "y", 
                Series = series, Subplots = subplots, 
                ScriptName = Path.Combine(MainClass.ScriptPath, title + ".py"),
                FigureName = Path.Combine(MainClass.FigurePath, title + ".pdf"),
                Python = MainClass.PythonPath
            };

            plotter.Plot();            
        }

        /// <summary>
        /// Plots the results.
        /// </summary>
        /// <returns>The results.</returns>
        /// <param name="numBases">Number bases.</param>
        /// <param name="dictionary">Dictionary.</param>
        /// <param name="coefficients">Coefficients.</param>
        /// <param name="subTitle">Sub title.</param>
        public static void PlotResults(int numBases, VectorGaussian[] dictionary, VectorGaussian[] coefficients, string subTitle = null)
        {
            PlotResults(
                numBases, 
                dictionary.Select(DistributionHelpers.IndependentApproximation).ToArray(),
                coefficients.Select(DistributionHelpers.IndependentApproximation).ToArray(),
                subTitle);
        }

        /// <summary>
        /// Plots the results.
        /// </summary>
        /// <returns>The results.</returns>
        /// <param name="numBases">Number bases.</param>
        /// <param name="dictionary">Dictionary.</param>
        /// <param name="coefficients">Coefficients.</param>
        /// <param name="subTitle">Sub title.</param>
        public static void PlotResults(int numBases, Gaussian[][] dictionary, Gaussian[][] coefficients, string subTitle = null)
        {
            for (var i = 0; i < numBases / 16; i++)
            {
                PlotPosteriors(dictionary, "Dictionary", subTitle, i * 16, 16, 4, 4, PlotType.ErrorLine);
            }

            if (numBases <= 32)
            {
                PlotPosteriors(coefficients, "Coefficients", subTitle, 0, 16, 4, 4, PlotType.Bar);
            }
            else
            {
                PlotPosteriors(coefficients, "Coefficients", subTitle, 0, 16, 8, 2, PlotType.Bar);
            }
        }


        public static void PlotResults(
            int numBases, int signalWidth,
            Gaussian[][] dictionary, Gaussian[][] coefficients,
            string st, bool isImage, bool plotDictionary, bool plotCoefficients)
        {
            const PlotType plotTypeDict = PlotType.ErrorLine;
            const PlotType plotTypeCoef = PlotType.ErrorBar;
            // const PlotType plotTypeDict = PlotType.Line;
            // const PlotType plotTypeCoef = PlotType.Bar;

            if (numBases < 16)
            {
                if (isImage)
                {
                    PlotImages(dictionary, signalWidth, "Dictionary", numBases, 1, plotDictionary);
                }
                else
                {
                    PlotPosteriors(dictionary, "Dictionary", st, 0, numBases, numBases, 1, plotTypeDict, plotDictionary);
                }
            }
            else
            {
                if (isImage)
                {
                    PlotImages(dictionary, signalWidth, "Dictionary", 4, 4, plotDictionary);
                }
                else
                {
                    PlotPosteriors(dictionary, "Dictionary", st, 0, 16, 4, 4, plotTypeDict, plotDictionary);
                }
            }

            if (coefficients != null)
            {
                PlotPosteriors(coefficients, "Coefficients", st, 0, 6, 3, 2, plotTypeCoef, plotCoefficients);

            }
        }

        /// <summary>
        /// Plots the reconstructions.
        /// </summary>
        /// <returns>The reconstructions.</returns>
        /// <param name="reconstructions">Reconstructions.</param>
        /// <param name="averageError">The average reconstruction error.</param>
        /// <param name="numToShow">Number to show.</param>
        /// <param name="rows">Rows.</param>
        /// <param name="cols">Cols.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="normalised">Whether these are normalised reconstructions.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void PlotReconstructions(Reconstruction[] reconstructions, double averageError, int numToShow,
            int rows, int cols, string subTitle = null, bool normalised = false, bool show = false)
        {
            var series1 = reconstructions.Take(numToShow).Select(
                (ia, i) => (ISeries)(new LineSeries
                {
                    Label = "signal",
                    X = ia.Signal,
                    Row = i/cols,
                    Column = i%cols
                })).ToArray();
            var series2 = reconstructions.Take(numToShow).Select(
                (ia, i) => (ISeries)(new ErrorLineSeries
                {
                    Label = "reconstruction",
                    ErrorLabel = "$\\pm$s.d.",
                    X = ia.Estimate.GetMeans(),
                    ErrorValues = ia.Estimate.GetStandardDeviations(),
                    Row = i/cols,
                    Column = i%cols
                })).ToArray();

            IList<ISeries> series = series1.Concat(series2).ToArray();

            string n = normalised ? " (normalised)" : string.Empty;

            // var series = new[] { new LineSeries { X = x1, Row = 0 }, new LineSeries { X = x2, Row = 1 } };
            var subplots = new Subplots {Rows = rows, Columns = cols, ShareX = true, ShareY = true};
            string sub = string.IsNullOrEmpty(subTitle) ? string.Empty : $"_{subTitle.Replace(" ", "_")}";
            var plotter = new Plotter
            {
                Title = $"Reconstructions{n}, RMSE={averageError:N4}",
                XLabel = "x",
                YLabel = "y",
                Series = series,
                Subplots = subplots,
                ScriptName = Path.Combine(MainClass.ScriptPath, $"Reconstructions_{n}{sub}.py"),
                FigureName = Path.Combine(MainClass.FigurePath, $"Reconstructions_{n}{sub}.pdf"),
                Python = MainClass.PythonPath,
                Show = show
            };
            plotter.Plot();
        }

        /// <summary>
        /// Plots the image reconstructions. Note that we assume the images are square
        /// </summary>
        /// <returns>The reconstructions.</returns>
        /// <param name="reconstructions">Reconstructions.</param>
        /// <param name="averageError">The average reconstruction error.</param>
        /// <param name="numToShow">Number to show.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="normalised">Whether these are normalised reconstructions.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void PlotImageReconstructions(Reconstruction[] reconstructions, double averageError, int numToShow,
            string subTitle = null, bool normalised = false, bool show = false)
        {
            var series1 = reconstructions.Take(numToShow).Select(
                (ia, i) => (ISeries) new HintonSeries
                {
                    Label = "signal",
                    Values = Data.Reshape(Vector.Build.Dense(ia.Signal)),
                    Row = i,
                    Column = 0
                }).ToArray();
            var series2 = reconstructions.Take(numToShow).Select(
                (ia, i) => (ISeries) new HintonSeries
                {
                    Label = "reconstruction",
                    // ErrorLabel = "$\\pm$s.d.",
                    Values = Data.Reshape(Vector.Build.Dense(ia.Estimate.GetMeans())),
                    // ErrorValues = ia.Estimate.GetStandardDeviations(),
                    Row = i,
                    Column = 1
                }).ToArray();

            IList<ISeries> series = series1.Concat(series2).ToArray();

            string n = normalised ? "(normalised)" : string.Empty;

            var subplots = new Subplots {Rows = numToShow, Columns = 2, ShareX = false, ShareY = false};
            string sub1 = string.IsNullOrEmpty(subTitle) ? string.Empty : $" {subTitle.Replace("_", " ")}";
            string sub2 = string.IsNullOrEmpty(subTitle) ? string.Empty : $"_{subTitle.Replace(" ", "_")}";
            string message = $"Reconstructions {n}{sub1}, avg. error={averageError:N4}";
            Console.WriteLine(message);
            var plotter = new Plotter
            {
                Title = message,
                XLabel = "x",
                YLabel = "y",
                Grid = false,
                Series = series,
                Subplots = subplots,
                ScriptName = Path.Combine(MainClass.ScriptPath, $"Reconstructions_{n}{sub2}.py"),
                FigureName = Path.Combine(MainClass.FigurePath, $"Reconstructions_{n}{sub2}.pdf"),
                Python = MainClass.PythonPath,
                Show = show
            };
            plotter.Plot();
        }

        // /// <summary>
        // /// Plots the reconstructions.
        // /// </summary>
        // /// <returns>The reconstructions.</returns>
        // /// <param name="signals">Signals.</param>
        // /// <param name="reconstructions">Reconstructions.</param>
        // /// <param name="title">Title.</param>
        // /// <param name="numToShow">Number to show.</param>
        // /// <param name="rows">Rows.</param>
        // /// <param name="cols">Cols.</param>
        // /// <param name="subTitle">Sub title.</param>
        // public static void PlotReconstructions(double[][] signals, double[][] reconstructions, string title, int numToShow, int rows, int cols, string subTitle = null)
        // {
        //     var r = signals.Zip(reconstructions, (s, e) => new Reconstruction { Signal = s, Estimate = e.Select(Gaussian.PointMass).ToArray() }).ToArray();
        //     PlotReconstructions(r, averageError, title, numToShow, rows, cols, subTitle);
        // }

        /// <summary>
        /// Plots the posteriors.
        /// </summary>
        /// <returns>The posteriors.</returns>
        /// <param name="posteriors">Posteriors.</param>
        /// <param name="title">Title.</param>
        /// <param name="subTitle">Sub title.</param>
        /// <param name="skip">Skip.</param>
        /// <param name="numToShow">Number to show.</param>
        /// <param name="rows">Rows.</param>
        /// <param name="cols">Cols.</param>
        /// <param name="plotType">Plot type.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void PlotPosteriors<T>(
            T[][] posteriors,
            string title,
            string subTitle = null,
            int skip = 0,
            int numToShow = 16, 
            int rows = 4, 
            int cols = 4, 
            PlotType plotType = PlotType.ErrorLine,
            bool show = true)
            where T : IDistribution<double>, CanGetMean<double>, CanGetVariance<double>
        {
            ISeries[] series;
            
            switch (plotType)
            {
                case PlotType.ErrorLine:
                    series = posteriors.Skip(skip).Take(numToShow).Select(
                        (ia, i) => (ISeries)(new ErrorLineSeries
                        {
                            // Label = i.ToString(),
                            X = ia.Select(x => x.GetMean()).ToArray(),
                            ErrorValues = ia.GetStandardDeviations(),
                            Row = i / cols,
                            Column = i % cols
                        })).ToArray();
                    break;
                case PlotType.Bar:
                    series = posteriors.Skip(skip).Take(numToShow).Select(
                        (ia, i) => (ISeries)(new BarSeries<string>
                        {
                            // Label = i.ToString(),
                            DependentValues = ia.GetMeans(),
                            Row = i / cols,
                            Column = i % cols
                        })).ToArray();
                    break;
                case PlotType.ErrorBar:
                    series = posteriors.Skip(skip).Take(numToShow).Select(
                        (ia, i) => (ISeries)(new BarSeries<string>
                        {
                            // Label = i.ToString(),
                            DependentValues = ia.GetMeans(),
                            ErrorValues = ia.GetStandardDeviations(),
                            Row = i / cols,
                            Column = i % cols
                        })).ToArray();
                    break;
                case PlotType.Line:
                    series = posteriors.Skip(skip).Take(numToShow).Select(
                        (ia, i) => (ISeries)(new LineSeries
                        {
                            // Label = i.ToString(),
                            X = ia.GetMeans(),
                            Row = i / cols,
                            Column = i % cols
                        })).ToArray();
                    break;
                default:
                    throw new ArgumentException("Unknonw plot type", nameof(plotType));
            }
            
            // var series = new[] { new LineSeries { X = x1, Row = 0 }, new LineSeries { X = x2, Row = 1 } };
            var subplots = new Subplots { Rows = rows, Columns = cols, ShareX = true, ShareY = true };
            string sub = string.IsNullOrEmpty(subTitle) ? string.Empty : $"_{subTitle.Replace(" ", "_")}";
            string sel = $"{skip}-{skip + numToShow}";
            var plotter = new Plotter 
            { 
                Title = $"{title} {sel}",
                XLabel = "x", YLabel = "y", Series = series, Subplots = subplots, 
                Python = MainClass.PythonPath,
                ScriptName = Path.Combine(MainClass.ScriptPath, $"{title}{sub}_{sel}.py"),
                FigureName = Path.Combine(MainClass.FigurePath, $"{title}{sub}_{sel}.pdf"),
                Show = show
            };
            
            plotter.Plot();
        }
        
        /// <summary>
        /// Plot errors with evidence on twinx
        /// </summary>
        public static void PlotErrorsWithEvidence(Results results, bool show = false)
        {
            // Here we're going to customise the Plotter.TwinPlot function
            var x = results.BasisCounts.Select(ia => (double)ia).ToArray();
            var y1 = results.Errors;
            var y2 = results.Evidence;
            const string title = "Effect of number of bases";
            const string xlabel = "#bases";
            const string y1Label = "Reconstruction error";
            const string y2Label = "Log Evidence";
            
            var series1 = new ISeries[] { new LineSeries { X = x, Y = y1, Color = "next(palette)", Label = "Reconstruction error" } };
            var series2 = new ISeries[] { new LineSeries { X = x, Y = y2, Color = "next(palette)", Label = "Evidence" } };

			// Here we build the plotting script for the second plot (without the pre/postamble), 
			// so we can append it to the script for the first plot
            var plotter2 = new Plotter { XLabel = xlabel, YLabel = y2Label, Series = series2, TwinX = true };
            plotter2.BuildScript();

            // TODO: http://matplotlib.org/examples/api/two_scales.html

            var plotter1 = new Plotter 
            { 
                Title = title, 
                XLabel = xlabel, 
                YLabel = y1Label, 
                Series = series1, 
                Python = MainClass.PythonPath,
                ScriptName = Path.Combine(MainClass.ScriptPath, "EffectOfBases"),
                FigureName = Path.Combine(MainClass.FigurePath, "EffectOfBases"),
                Show = show
            };
            
            plotter1.Plot(plotter2.Script);
            
        }

        /// <summary>
        /// Plots the image.
        /// </summary>
        /// <param name="imageFlat">Image flat.</param>
        /// <param name="show">Whether to show the plot.</param>
        public static void PlotImage(Vector imageFlat, bool show = false)
		{
			
			// DenseOfColumnMajor(rows, columns, m.Row(0));
			// Plotter.Hinton(image);
			var plotter = new Plotter 
            { 
                Series = new ISeries[] { new MatrixSeries { Values = Data.Reshape(imageFlat) } }, 
                Grid = false,
                Python = MainClass.PythonPath,
                ScriptName = Path.Combine(MainClass.ScriptPath, "EffectOfBases"),
                FigureName = Path.Combine(MainClass.FigurePath, "EffectOfBases"),
                Show = show
            };
			plotter.Plot();
		}

        /// <summary>
        /// Plots the images.
        /// </summary>
        /// <param name="imagesFlat">Images.</param>
        /// <param name="title">The title.</param>
        /// <param name="rows">Number of rows.</param>
        /// <param name="columns">Number of columns.</param>
        /// <param name="show">Whether to show the plot</param>
        public static void PlotImages(Matrix imagesFlat, string title, int rows, int columns, bool show = false)
		{
			var series =
			    (from t in imagesFlat.EnumerateRowsIndexed()
			     let index = t.Item1 let image = Data.Reshape(t.Item2)
			     select new MatrixSeries { Values = image, Row = index / columns, Column = index % columns }
                ).Cast<ISeries>().ToList();

            var plotter = new Plotter
            {
                Title = title,
                Series = series, 
                Grid = false, 
                Subplots = new Subplots { Rows = rows, Columns = columns }, 
                Python = MainClass.PythonPath,
                ScriptName = Path.Combine(MainClass.ScriptPath, "EffectOfBases"),
                FigureName = Path.Combine(MainClass.FigurePath, "EffectOfBases"),
                Show = show 
            };
			plotter.Plot();
		}
        
        public static void PlotImages(Gaussian[][] images, int imageWidth, string title, int rows, int columns, bool show)
        {
            var dictionary = Matrix.Build.DenseOfRowArrays(images.GetMeans<Gaussian>());
            PlotImages(dictionary.SubMatrix(0, rows * columns, 0, imageWidth), title, rows, columns, show);
        }
    }
}