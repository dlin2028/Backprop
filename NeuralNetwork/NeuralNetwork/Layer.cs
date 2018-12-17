using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Layer
    {
        public Neuron[] Neurons;
        public double[] Output => Neurons.Select(n => n.Output).ToArray(); //Linq expressions help with basic selections

        /// <summary>
        /// Constructs a Layer object
        /// </summary>
        /// <param name="activation">the activation function to be set in all neurons of the constructed layer</param>
        /// <param name="inputCount">the number of inputs of the layer</param>
        /// <param name="neuronCount">the number of neurons in the layer</param>
        public Layer(IActivation activation, int inputCount, int neuronCount)
        {
            Neurons = new Neuron[neuronCount];
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = new Neuron(activation, inputCount);
            }
        }

        /// <summary>
        /// Randomizes each neuron's weights and biases in the layer
        /// </summary>
        /// <param name="rand">a given random number generator in case seeds are wanted to control the randomization process</param>
        public void Randomize(Random rand)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].RandomizeWeights(rand);
            }
        }

        /// <summary>
        /// Computes the result for each neuron in the layer based of the given input
        /// </summary>
        /// <param name="input">the given input used to compute the output for each neuron</param>
        /// <returns>the outputs of each neuron stored in a double array</returns>
        public double[] Compute(double[] input)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i].Compute(input);
            }
            return Output;
        }
    }
}

