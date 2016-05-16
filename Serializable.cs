//
// Serializable.cs
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

    public class Serializable<T>
    {
        public string Name { get; set; }

        public string GetJson()
        {
            return JsonConvert.SerializeObject(this, Formatting.Indented,
                new JsonSerializerSettings {NullValueHandling = NullValueHandling.Ignore});
        }

        public void Save(string path)
        {
            // Save results to json
            // string now = DateTime.Now.ToString("yyyy-dd-M--HH-mm-ss");
            // string filename = Path.Combine(path, $"{GetType()}_{Name}_{now}.json");
            string filename = Path.Combine(path, $"{GetType()}_{Name.Replace("  ", "_")}.json");
            Console.WriteLine($"Saving results to {filename}");
            File.WriteAllText(filename, GetJson());
        }

        public static T Load(string filename)
        {
            Console.WriteLine($"Loading type {typeof(T).Name}: {filename}");
            var obj = JsonConvert.DeserializeObject<T>(File.ReadAllText(filename));
            return obj;
        }
    }
}