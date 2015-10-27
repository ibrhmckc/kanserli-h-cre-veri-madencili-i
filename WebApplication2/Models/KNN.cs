using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System;
using System.Linq;
using Accord.Math;
using Accord.MachineLearning.Structures;
namespace WebApplication2.Models
{
    public class KNearestNeighborsss
    {

        int ClassCount ;

    
    

        private KDTree<int> tree;

    
        /// 
        public KNearestNeighborsss(int k, double[][] inputs, int[] outputs)
             //base(k, inputs, outputs, Accord.Math.Distance.Euclidean)
        {
            this.tree = KDTree.FromData(inputs, outputs);
        }

      
        /// 
        public KNearestNeighborsss(int k, int classes, double[][] inputs, int[] outputs)
             //base(k, classes, inputs, outputs, Accord.Math.Distance.Euclidean)
        {
            this.tree = KDTree.FromData(inputs, outputs);
        }

        public KNearestNeighborsss(int k, int classes, double[][] inputs, int[] outputs, Func<double[], double[], double> distance)
             //base(k, classes, inputs, outputs, distance)
        {
            this.tree = KDTree.FromData(inputs, outputs, distance);
        }


        public  int Compute(double[] input, out double[] scores)
        {
            int K=0;
            KDTreeNodeCollection<int> neighbors = tree.Nearest(input,K);

            scores = new double[ClassCount];

            foreach (var point in neighbors)
            {
                int label = point.Node.Value;
                double d = point.Distance;

                // Convert to similarity measure
                scores[label] += 1.0 / (1.0 + d);
            }

            // Get the maximum weighted score
            int result; scores.Max(out result);

            return result;
        }


    }

}

    
