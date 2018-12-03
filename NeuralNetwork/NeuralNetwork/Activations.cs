using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public static class Activations
    {

        /// <summary>
        /// The basic activation function used by perceptrons and simple neural networks.
        /// </summary>
        /// <param name="x">the input to activate</param>
        /// <returns>returns 0 if x is less than 0. otherwise 1</returns>
        public static double BinaryStep(double x)
        {
            return x < 0.5 ? 0 : 1;
        }

        /// <summary>
        /// Example of a more complex activation function. Will be used later for backpropogation since it can be derived.
        /// </summary>
        /// <param name="x">the input to activate</param>
        /// <returns>1 / (1 + e^-x)</returns>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }

}
