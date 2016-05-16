//
// ResultsCollection.cs
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
    using System.IO;
    using Newtonsoft.Json;
    using System.Collections.Generic;

    public class ResultsCollection
    {
        public int NumTrain { get; set; }

        public int NumTest { get; set; }

        public string Name { get; set; }

        public bool Normalised { get; set; }

        public IList<int> Bases { get; set; }

        public IList<Results> Results { get; set; } = new List<Results>();

        public void Save(string path)
        {
            // Save results to json
            string json = JsonConvert.SerializeObject(this, Formatting.Indented);
            string now = DateTime.Now.ToString("yyyy-dd-M--HH-mm-ss");
            string filename = Path.Combine(path, $"results_{Name}_{now}.json");
            Console.WriteLine($"Saving results to {filename}");
            File.WriteAllText(filename, json);
        }
    }
}