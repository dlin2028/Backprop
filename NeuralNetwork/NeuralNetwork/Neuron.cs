using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetwork
{
    public class Neuron
    {
        //Standard Use
        public double BiasWeight;
        public double[] Weights;
        public IActivation Activation;
        public double Input;
        public double Output;

        //Backprop Variables
        public double PartialDerivative;
        public double BiasUpdate;
        public double[] WeightUpdates;

        public double PrevBiasUpdate;
        public double[] PrevWeightUpdates;

        public Neuron(IActivation activation, int numberOfInputs)
        {
            this.Activation = activation;
            Weights = new double[numberOfInputs];
            WeightUpdates = new double[numberOfInputs];
            PrevWeightUpdates = new double[numberOfInputs];
        }

        public Neuron Clone()
        {
            var output = new Neuron(Activation, Weights.Length);

            output.Weights = new double[Weights.Length];
            Array.Copy(Weights, output.Weights, Weights.Length);
            output.BiasWeight = BiasWeight;
            output.Output = Output;
            output.BiasUpdate = BiasUpdate;
            output.WeightUpdates = WeightUpdates;
            output.PartialDerivative = PartialDerivative;

            return output;
        }

        public void RandomizeWeights(Random rng)
        {
            BiasWeight = rng.NextDouble(-0.5, 0.5);
            for (int i = 0; i < Weights.Length; i++)
            {
                Weights[i] = rng.NextDouble(-0.5, 0.5);
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

            Input = dotProduct + BiasWeight;
            Output = Activation.Function(Input);

            return Output;
        }
    }
}
