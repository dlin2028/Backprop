using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public static class Activations
    {
        public static IActivation Sigmoid = new Sigmoid();
        public static IActivation BinaryStep = new BinaryStep();
        public static IActivation Tanh = new Tanh();
        public static IActivation Identity = new Identity();
    }

    public class Sigmoid : IActivation
    {
        public double Function(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public double Derivative(double input)
        {
            return Function(input) * (1 - Function(input));
        }
    }
    
    public class BinaryStep : IActivation
    {
        public double Function(double input)
        {
            return input > 0.5 ? 1 : 0;
        }

        public double Derivative(double input)
        {
            return 0;
        }
    }

    public class Tanh : IActivation
    {
        public double Function(double input)
        {
            return Math.Tanh(input);
        }

        public double Derivative(double input)
        {
            return 1 - (Function(input) * Function(input));
        }
    }

    public class Identity : IActivation
    {
        public double Function(double input)
        {
            return input;
        }

        public double Derivative(double input)
        {
            return 1;
        }
    }
}
