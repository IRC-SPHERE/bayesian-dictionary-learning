//
// IModel.cs
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
//
// SparseDLYang.cs
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
    using MicrosoftResearch.Infer;

    /// <summary>
    /// The inference mode
    /// </summary>
    public enum Mode
    {
        Train,
        TrainFixed,
        TrainOnline,
        Reconstruct,
    }

    /// <summary>
    /// Model interface
    /// </summary>
    public interface IModel
    {
        /// <summary>
        /// Model parameters
        /// </summary>
        BDLParameters Parameters { get; set; }

        /// <summary>
        /// Constructs the model
        /// </summary>
        void ConstructModel();

        /// <summary>
        /// Train the multiclass model.
        /// </summary>
        /// <param name="priors">The priors.</param>
        /// <param name="signals">The input signals.</param>
        Marginals Train(Marginals priors, double[][] signals);

        /// <summary>
        /// Reconstruct using the dictionary and coefficients provided.
        /// </summary>
        /// <param name="priors">The priors (dictionary, coefficients, noise precision.</param>
        Marginals Reconstruct(Marginals priors);

        /// <summary>
        /// Add a handler for when an inference update happens.
        /// </summary>
        /// <param name="handler">The event handler.</param>
        void AddUpdateHandler(EventHandler<ProgressChangedEventArgs> handler);
    }

    /// <summary>
    /// Batch model dummy interface.
    /// </summary>
    public interface IBatchModel { }

    /// <summary>
    /// Vector Gaussian model dummy interface.
    /// </summary>
    public interface IVectorModel { }
}