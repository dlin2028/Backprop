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

        /// <summary>
        /// Mutates the weights and biases of the neural network to slightly alter it based on a given mutation rate
        /// </summary>
        /// <param name="random">a given random number generator in case seeds are wanted to control the randomization process</param>
        /// <param name="rate">the given mutation rate. ranged 0-1, representing the % of the neural network that will be mutated</param>
        public void Mutate(Random random, double rate)
        {
            oldLayers = new List<Layer>(Layers.ToArray());
            foreach (Layer layer in Layers)
            {
                foreach (Neuron perceptron in layer.Neurons)
                {
                    if (random.NextDouble() < rate)
                    {
                        //a percentage based change is ideal since it allows smaller numbers to have small changes
                        perceptron.BiasWeight *= random.NextDouble(0.5, 1.5) * random.RandomSign();
                    }

                    for (int w = 0; w < perceptron.Weights.Length; w++)
                    {
                        if (random.NextDouble() < rate)
                        {
                            //a percentage based change is ideal since it allows smaller numbers to have small changes
                            perceptron.Weights[w] *= random.NextDouble(0.5, 1.5) * random.RandomSign();
                        }
                    }
                }
            }
        }

        //Crossover
        public void Crossover(NeuralNet other, Random rng)
        {
            for (int i = 0; i < Layers.Count; i++)
            {
                for (int j = 0; j < Layers[i].Neurons.Length; j++)
                {
                    if(rng.Next(2) == 0)
                    {
                        Layers[i].Neurons[j].BiasWeight = other.Layers[i].Neurons[j].BiasWeight;
                        other.Layers[i].Neurons[j].Weights.CopyTo(Layers[i].Neurons[j].Weights, 0);
                    }
                }
            }
        }


        //Backprop
        //ClearUpdates: sets PartialDerivative, BiasUpdate, WeightUpdats all to ZERO
        //CalculateError: finds partial derivative for each neuron
        //CalculateUpdate: finds bias & weight updates for each neuron
        //ApplyUpdate: add the bias and weight updates to the dendrites


        public void CalculateError()
        {
            Layer outputLayer = Layers[Layers.Count - 1];
            foreach (var neuron in outputLayer.Neurons)
            {
            }
        }

        public void ClearUpdates()
        {
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuron.PartialDerivative = 0;
                    neuron.biasUpdate = 0;
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
                    neuron.BiasWeight += neuron.biasUpdate;
                    for (int i = 0; i < neuron.Weights.Length; i++)
                    {
                        neuron.Weights[i] += neuron.WeightUpdates[i];
                    }
                }
            }
        }

    }
}
