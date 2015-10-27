using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using AForge;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace WebApplication2.Models
{
    public class DecssionsTreee
    {


 
    public class ID3Learning
    {

        private DecisionTree tree;

        private int maxHeight;
        private IntRange[] inputRanges;
        private int outputClasses;

        private bool[] attributes;


      
        public int MaxHeight
        {
            get { return maxHeight; }
            set
            {
                if (maxHeight <= 0 || maxHeight > attributes.Length)
                    throw new ArgumentOutOfRangeException("deger",
                        "agaç nodları oluşturulamadı.");
                maxHeight = value;
            }
        }


        /// <summary>
        ///   Creates a new ID3 learning algorithm.
        /// </summary>
        /// 
        /// <param name="tree">The decision tree to be generated.</param>
        /// 
        public ID3Learning(DecisionTree tree)
        {
            this.tree = tree;
            this.inputRanges = new IntRange[tree.InputCount];
            this.outputClasses = tree.OutputClasses;
            this.attributes = new bool[tree.InputCount];
            this.maxHeight = attributes.Length;

            for (int i = 0; i < tree.Attributes.Count; i++)
                if (tree.Attributes[i].Nature != DecisionVariableKind.Discrete)
                    throw new ArgumentException("id3 inputları hatalı geldi.");

            for (int i = 0; i < inputRanges.Length; i++)
                inputRanges[i] = tree.Attributes[i].Range.ToIntRange(false);
        }


        
        public double Run(int[][] inputs, int[] outputs)
        {
            // Reset the usage of all attributes
            for (int i = 0; i < attributes.Length; i++)
                attributes[i] = false;

            // 1. Create a root node for the tree
            this.tree.Root = new DecisionNode(tree);

            split(tree.Root, inputs, outputs);

            // Return the classification error
            return ComputeError(inputs, outputs);
        }

      
        public double ComputeError(int[][] inputs, int[] outputs)
        {
            int miss = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                //if (tree.Compute(inputs[i].ToDouble()) != outputs[i])
                    miss++;
            }

            return (double)miss / inputs.Length;
        }

        private void split(DecisionNode root, int[][] input, int[] output)
        {

            double entropy = Accord.Statistics.Tools.Entropy(output, outputClasses);

            if (entropy == 0)
            {
                if (output.Length > 0)
                    root.Output = output[0];
                return;
            }

          
            int predictors = attributes.Count(x => x == false);

            if (predictors < attributes.Length - maxHeight)
            {
                root.Output = Accord.Statistics.Tools.Mode(output);
                return;
            }



            double[] scores = new double[predictors];
            double[] entropies = new double[predictors];
            int[][][] partitions = new int[predictors][][];

            // Retrieve candidate attribute indices
            int[] candidates = new int[predictors];
            for (int i = 0, k = 0; i < attributes.Length; i++)
                if (!attributes[i]) candidates[k++] = i;


            // For each attribute in the data set
#if SERIAL
            for (int i = 0; i < scores.Length; i++)
#else
            Parallel.For(0, scores.Length, i =>
#endif
            {
                scores[i] = computeGainRatio(input, output, candidates[i],
                    entropy, out partitions[i]);
            }
#if !SERIAL
);
#endif

            // Select the attribute with maximum gain ratio
            int maxGainIndex=44; //scores.Max(out maxGainIndex);
            var maxGainPartition = partitions[maxGainIndex];
            var maxGainEntropy = entropies[maxGainIndex];
            var maxGainAttribute = candidates[maxGainIndex];
            var maxGainRange = inputRanges[maxGainAttribute];

            attributes[maxGainAttribute] = true;

            DecisionNode[] children = new DecisionNode[maxGainPartition.Length];

            for (int i = 0; i < children.Length; i++)
            {
                children[i] = new DecisionNode(tree);
                children[i].Parent = root;
                children[i].Comparison = ComparisonKind.Equal;
                children[i].Value = i + maxGainRange.Min;


                int[][] inputSubset = input;
                int[] outputSubset = output;

                split(children[i], inputSubset, outputSubset); // recursion
            }


            attributes[maxGainAttribute] = false;

            root.Branches.AttributeIndex = maxGainAttribute;
            root.Branches.AddRange(children);
        }


        private double computeInfo(int[][] input, int[] output, int attributeIndex, out int[][] partitions)
        {
            double info = 0;

            IntRange valueRange = inputRanges[attributeIndex];
            partitions = new int[valueRange.Length + 1][];

            // For each possible value of the attribute
            for (int i = 0; i < partitions.Length; i++)
            {
                int value = valueRange.Min + i;

            
                int[] outputSubset = output;

                // Check the entropy gain originating from this partitioning
                double e = Accord.Statistics.Tools.Entropy(outputSubset, outputClasses);

                info += ((double)outputSubset.Length / output.Length) * e;
            }

            return info;
        }

        private double computeInfoGain(int[][] input, int[] output, int attributeIndex,
            double entropy, out int[][] partitions)
        {
            return entropy - computeInfo(input, output, attributeIndex, out partitions);
        }

        private double computeGainRatio(int[][] input, int[] output, int attributeIndex,
            double entropy, out int[][] partitions)
        {
            double infoGain = computeInfoGain(input, output, attributeIndex, entropy, out partitions);
            double splitInfo = Measures.SplitInformation(output.Length, partitions);

            return infoGain == 0 ? 0 : infoGain / splitInfo;
        }
    }
}

    }
