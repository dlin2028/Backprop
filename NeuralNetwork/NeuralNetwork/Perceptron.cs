using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    class Neuron
    {
        //Standard Use
        public double BiasWeight;
        public double[] Weights;
        public Func<double, double> activation;
        public double Output;

        //Backprop Variables
        public double PartialDerivative;
        public double BiasUpdate;
        public double[] WeightUpdates;

        public Neuron(Func<double, double> activation, int numberOfInputs)
        {
            this.activation = activation;
            Weights = new double[numberOfInputs];
            WeightUpdates = new double[numberOfInputs];
        }

        public void RandomizeWeights(Random rng)
        {
            BiasWeight = rng.NextDouble();
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rng.NextDouble();
            }
        }

        public double Compute(double[] inputs)
        {
            if (inputs.Length != Weights.Length)
            {
                throw new Exception("wrong number of inputs");
            }

            double dotProduct = 0;
            for (int i = 0; i < Weights.Length; i++)
            {
                dotProduct += Weights[i] * inputs[i];
            }

            Output = activation(dotProduct);
            return Output;
        }


        public void Train(double[] inputs, double desiredOutput)
        {
            double output = Compute(inputs);
            double error = desiredOutput - output;
            for (int i = 0; i < inputs.Length; i++)
            {
                Weights[i] += error * inputs[i];
            }
            BiasWeight += error;
        }

    }
}
