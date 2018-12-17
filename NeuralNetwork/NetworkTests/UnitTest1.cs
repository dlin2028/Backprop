using System;
using Xunit;
using NeuralNetwork;
using System.Collections.Generic;

namespace NetworkTests
{
    public class NetworkTests
    {
        [Fact]
        public void XOR_Backprop()
        {
            int seed = Guid.NewGuid().GetHashCode();
            Random rand = new Random(seed);

            List<Layer> layers = new List<Layer>();
            layers.Add(new Layer(Activations.Sigmoid, 2, 2));
            layers.Add(new Layer(Activations.Sigmoid, 2, 2));
            layers.Add(new Layer(Activations.Sigmoid, 2, 1));

            NeuralNet net = new NeuralNet(layers.ToArray());
            net.Randomize(rand);

            double[][] inputs = {
                new double[]{ 0, 0 },
                new double[]{ 0, 1 },
                new double[]{ 1, 0 },
                new double[]{ 1, 1 }
            };
            double[][] outputs = {
                new double[]{ 0 },
                new double[]{ 1 },
                new double[]{ 1 },
                new double[]{ 0 }
            };

            int epochs = 0;

            double error = 1;
            while (error > 0)
            {
                net.Backprop(inputs, outputs, 0.9);

                //net.Compute(inputs)
                //perform  binary step on outputs
                //calculate MAE with those new outputs
                error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    error += Activations.BinaryStep.Function(net.Compute(inputs[i])[0]) - outputs[i][0];
                }
                Console.SetCursorPosition(0, 0);


                Console.WriteLine($"0 ^ 0 = {Activations.BinaryStep.Function(net.Compute(inputs[0])[0])}");
                Console.WriteLine($"0 ^ 1 = {Activations.BinaryStep.Function(net.Compute(inputs[1])[0])}");
                Console.WriteLine($"1 ^ 0 = {Activations.BinaryStep.Function(net.Compute(inputs[2])[0])}");
                Console.WriteLine($"1 ^ 1 = {Activations.BinaryStep.Function(net.Compute(inputs[3])[0])}");


                Console.Write($"{error:#.0000}");
                epochs++;
            }
        }
    }
}
