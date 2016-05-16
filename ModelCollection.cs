//
// ModelCollection.cs
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
    using Models;
//    using Model = Models.BDLOnline;
//    using Model = Models.BDL;
//    using OnlineModel = Models.BDLOnline;
    using Model = Models.BDLSimple;
//    using OnlineModel = Models.BDLSimple;
//    using Model = Models.BDLSpikeSlab;

    public class ModelCollection
    {
        public IModel Train { get; set; } //= new Model(Mode.Train, false, true, false, false, false, true);
//        public IModel TrainOnline { get; set; } //= new OnlineModel(Mode.TrainOnline, false, true, false, false, false, true);
        public IModel TrainFixed { get; set; } //= new Model(Mode.TrainFixed, false, true, false, false, false, true);
//        public IModel TrainFixedOnline { get; set; } //= new OnlineModel(Mode.TrainOnline, false, true, false, false, false, true);
        public IModel Reconstruct { get; set; } //= new Model(Mode.Reconstruct, false, true, false, false, false, true);
//        public IModel ReconstructOnline { get; set; } //= new OnlineModel(Mode.Reconstruct, false, true, false, false, false, true);

        public static ModelCollection CreateModels(bool sparse = true, bool normConstraints = false, bool includeBias = false, bool missingData = false)
        {
            return new ModelCollection
            {
                Train =
                    new Model(
                        new BDLParameters
                        {
                            Mode = Mode.Train,
                            Sparse = sparse,
                            NormConstraints = normConstraints,
                            IncludeBias = includeBias
                        }, true),
                TrainFixed =
                    new Model(
                        new BDLParameters
                        {
                            Mode = Mode.TrainFixed,
                            Sparse = sparse,
                            NormConstraints = normConstraints,
                            IncludeBias = includeBias
                        }, true),
                Reconstruct =
                    new Model(
                        new BDLParameters
                        {
                            Mode = Mode.Reconstruct,
                            Sparse = sparse,
                            NormConstraints = normConstraints,
                            IncludeBias = includeBias
                        }, true),
            };
        }
    }
}