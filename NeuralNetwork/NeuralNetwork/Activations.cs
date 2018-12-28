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

        public double Inverse(double input)
        {
            return Math.Log(input, Math.E);
        }

        public double Derivative(double input)
        {
             return input * (1 - input);
        }
    }
    
    public class BinaryStep : IActivation
    {
        public double Function(double input)
        {
            return input > 0.5 ? 1 : 0;
        }

        public double Inverse(double input)
        {
            throw new NotImplementedException();
        }

        public double Derivative(double input)
        {
            return input * (1 - input);
        }
    }

    public class Tanh : IActivation
    {
        public double Function(double input)
        {
            return Math.Tanh(input);
        }

        public double Inverse(double input)
        {
            throw new NotImplementedException();
        }

        public double Derivative(double input)
        {
            return 1 - input * input;
        }
    }

    public class Identity : IActivation
    {
        public double Function(double input)
        {
            return input;
        }

        public double Inverse(double input)
        {
            return input;
        }

        public double Derivative(double input)
        {
            return 1;
        }
    }
}
