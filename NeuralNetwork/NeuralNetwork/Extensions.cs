using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{
    public static class Extensions
    {
        public static double NextDouble(this Random sender, double min, double max)
        {
            return sender.NextDouble() * (max - min) + min;
        }

        public static int RandomSign(this Random sender)
        {
            return sender.Next(0, 2) * 2 - 1; //no branching required!
        }

        public static T Clamp<T>(this T val, T min, T max) where T : IComparable<T>
        {
            if (val.CompareTo(min) < 0) return min;
            else if (val.CompareTo(max) > 0) return max;
            else return val;
        }
    }
}
