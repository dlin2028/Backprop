using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public interface IActivation
    {
        double Function(double input);
        double Derivative(double input);
    }
}
