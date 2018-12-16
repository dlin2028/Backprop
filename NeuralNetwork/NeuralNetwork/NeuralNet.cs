using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class NeuralNet
    {
        List<Layer> Layers;
        public double[] Output => Layers.Last().Output;

        private List<Layer> oldLayers;

        /// <summary>
        /// Constructs a Feed Forward Neural Network object
        /// </summary>
        /// <param name="activation">the activation function used for computation by the neurons</param>
        /// <param name="inputCount">the number of inputs to the neural network</param>
        /// <param name="layerNeurons">an array representing how many neurons are in each of the hidden layers and output layer of the neural network</param>
        public NeuralNet(Func<double, double> activation, int inputCount, params int[] layerNeurons)
        {
            Layers = new List<Layer>
            {
                new Layer(activation, inputCount, layerNeurons[0])
            };
            for (int i = 1; i < layerNeurons.Length; i++)
            {
                //the number of inputs of a layer is the number of outputs of the previous layer
                Layers.Add(new Layer(activation, Layers[i - 1].Neurons.Length, layerNeurons[i]));
            }
        }

        public double[] Compute(double[] data, int layer = 0)
        {
            if (layer == Layers.Count - 1)
            {
                return Layers[layer].Compute(data);
            }
            return Compute(Layers[layer].Compute(data), layer + 1);
        }
        
        public double Fitness(double[][] inputs, double[][] desiredOutputs)
        {
            double fitness = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                var input = inputs[i];
                var desiredOutput = desiredOutputs[i];
                for (int j = 0; j < input.Length; j++)
                {
                    fitness += Math.Abs(Compute(input)[j] - desiredOutput[0]);
                }
            }
            return fitness / inputs.Length;
        }

        /// <summary>
        /// Randomizes the weights and biases of the neural network
        /// </summary>
        /// <param name="rand">a given random number generator in case seeds are wanted to control the randomization process</param>
        public void Randomize(Random rand)
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                Layers[i].Randomize(rand);
            }
        }


        //Backprop
        //ClearUpdates: sets PartialDerivative, BiasUpdate, WeightUpdats all to ZERO
        //CalculateError: finds partial derivative for each neuron
        //CalculateUpdate: finds bias & weight updates for each neuron
        //ApplyUpdate: add the bias and weight updates to the dendrites

        public void Backprop(double[][] inputs, double[][] desiredOutputs, double learningRate)
        {
            //clear the existing deltas in preperation of the algorithm
            ClearUpdates();
            for (int i = 0; i < inputs.Length; i++)
            {
                //train the net for each row of test data. creating a summation of weight & bias changes per test
                Compute(inputs[i]);
                CalculateError(desiredOutputs[i]);
                CalculateUpdates(inputs[i], learningRate);
            }
            //apply the weight & bias changes after training on all of the given test data


            ApplyUpdates();     //why is this not in the loop? Wouldn't it just apply the update from the last set of input/outputs?
        }

        public void CalculateError(double[] desiredOutputs)
        {
            Layer outputLayer = Layers[Layers.Count - 1];
            for (int i = 0; i < desiredOutputs.Length; i++)
            {
                Neuron neuron = outputLayer.Neurons[i];
                double error = neuron.Output - desiredOutputs[i];

                neuron.PartialDerivative = error * (error * (1 - error)); //I should probably fix this later
            }

            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                Layer currLayer = Layers[i];
                Layer nextLayer = Layers[i + 1];

                for (int j = 0; j < currLayer.Neurons.Length; j++)
                {
                    Neuron neuron = currLayer.Neurons[j];


                    double error = 0.0;
                    foreach (Neuron nextNeuron in nextLayer.Neurons)
                    {
                        error += nextNeuron.PartialDerivative * nextNeuron.Weights[j];
                    }
                    neuron.PartialDerivative = error * (error * (1 - error)); //this too
                }
            }
        }

        private void CalculateUpdates(double[] input, double learningRate)
        {
            //Input Layer
            Layer inputLayer = Layers[0];
            for (int i = 0; i < inputLayer.Neurons.Length; i++)
            {
                Neuron neuron = inputLayer.Neurons[i];
                for (int j = 0; j < neuron.Weights.Length; j++)
                {
                    neuron.WeightUpdates[j] += learningRate * neuron.PartialDerivative * input[j];
                }
                neuron.BiasUpdate += learningRate * neuron.PartialDerivative;
            }

            //Hidden Layers
            for (int i = 1; i < Layers.Count; i++)
            {
                Layer currLayer = Layers[i];
                Layer prevLayer = Layers[i - 1];

                for (int j = 0; j < currLayer.Neurons.Length; j++)
                {
                    Neuron neuron = currLayer.Neurons[j];
                    for (int k = 0; k < neuron.Weights.Length; k++)
                    {
                        neuron.WeightUpdates[k] += learningRate * neuron.PartialDerivative * prevLayer.Output[k];
                    }
                    neuron.BiasUpdate += learningRate * neuron.PartialDerivative;
                }
            }
        }

        public void ClearUpdates()
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.PartialDerivative = 0;
                    neuron.BiasUpdate = 0;
                    for (int i = 0; i < neuron.Weights.Length; i++)
                    {
                        neuron.WeightUpdates[i] = 0;
                    }
                }
            }
        }

        public void ApplyUpdates(bool clear = false)
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.BiasWeight += neuron.BiasUpdate;
                    for (int i = 0; i < neuron.Weights.Length; i++)
                    {
                        neuron.Weights[i] += neuron.WeightUpdates[i];
                    }
                }
            }
        }

    }
}
