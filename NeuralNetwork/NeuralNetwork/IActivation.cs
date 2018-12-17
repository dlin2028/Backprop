using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public interface IActivation
    {
        double Function(double input);
        double Inverse(double output);
        double Derivative(double input);
    }
}
