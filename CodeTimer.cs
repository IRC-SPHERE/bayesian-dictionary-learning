//
// CodeTimer.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2015 University of Bristol
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
    using System.Diagnostics;

	/// <summary>
	/// The code timer. Can be used as follows:
	///  using (new CodeTimer("Some message"))
	///  {
	///      // Do some stuff
	///  }
	/// </summary>
	public class CodeTimer : IDisposable
	{
		/// <summary>
		/// The stopwatch.
		/// </summary>
		private readonly Stopwatch stopwatch = new Stopwatch();

		/// <summary>
		/// Initializes a new instance of the <see cref="CodeTimer"/> class.
		/// </summary>
		/// <param name="message">The message.</param>
		public CodeTimer(string message)
		{
			Console.Write(message + "...");
			this.stopwatch.Start();
		}

		/// <summary>
		/// Times the function.
		/// </summary>
		/// <typeparam name="TInput">The type of the input.</typeparam>
		/// <typeparam name="TOutput">The type of the output.</typeparam>
		/// <param name="message">The message.</param>
		/// <param name="func">The function.</param>
		/// <param name="p">The p.</param>
		/// <returns>The function output</returns>
		public static TOutput TimeFunction<TInput, TOutput>(string message, Func<TInput, TOutput> func, TInput p)
		{
			TOutput ret;
			using (new CodeTimer(message))
			{
				ret = func(p);
			}

			return ret;
		}

		/// <summary>
		/// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
		/// </summary>
		public void Dispose()
		{
			this.stopwatch.Stop();
		    Console.Write(
		        $" done. (elapsed = {((double) this.stopwatch.ElapsedMilliseconds/1000).ToString("#0.00")}s)\n");
		}
	}
}

