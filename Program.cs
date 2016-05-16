//
// Program.cs
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

    /// <summary>
    /// Main class.
    /// </summary>
    public class MainClass
    {
        public const string ScriptPath = "../../scripts";
        public const string FigurePath = "../../figures";
        public const string ResultPath = "../../results";
        public const string MarginalPath = "../../marginals";
        public const string PythonPath = "/usr/bin/python";

        /// <summary>
        /// The entry point of the program, where the program control starts and ends.
        /// </summary>
        /// <param name="args">The command-line arguments.</param>
        public static void Main(string[] args)
        {
            var funcs = new Action[]
            {
                Runner.Toy,
                Runner.AccelerometerMain,
                Runner.EffectOfBases,
                Runner.Mnist,
                Runner.Online,
                Runner.Convergence,
                Runner.Missing,
                Runner.AcceleromterSphere
            }.Select(
                (ia, i) => new {ia, i})
                .ToDictionary(f => f.ia, flag => args.Length > flag.i && args[flag.i] == "1");

            foreach (var f in funcs.Where(f => f.Value))
            {
                using (new CodeTimer(f.Key.Method.Name))
                {
                    f.Key();
                }
            }
        }
    }
}
