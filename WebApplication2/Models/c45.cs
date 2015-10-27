using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using AForge;
using System;
  using System;
    using System.Linq;
    using Accord.Math;
    using AForge;
    using Parallel = System.Threading.Tasks.Parallel;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace WebApplication2.Models
{
    public class c45
    {

 
   


        private DecisionTree tree;

        private int maxHeight;
        private double[][] thresholds;
        private IntRange[] inputRanges;
        private int outputClasses;

        private bool[] attributes;


        /// <summary>
        ///   Gets or sets the maximum allowed 
        ///   height when learning a tree.
        /// </summary>
        /// 
        public int MaxHeight
        {
            get { return maxHeight; }
            set
            {
                if (maxHeight <= 0 || maxHeight > attributes.Length)
                    throw new ArgumentOutOfRangeException("value", 
                        "The height must be greater than zero and less than the number of variables in the tree.");
                maxHeight = value;
            }
        }


        /// <summary>
        ///   Creates a new C4.5 learning algorithm.
        /// </summary>
        /// 
        /// <param name="tree">The decision tree to be generated.</param>
        /// 
        public c45(DecisionTree tree)
        {
            this.tree = tree;
            this.attributes = new bool[tree.InputCount];
            this.inputRanges = new IntRange[tree.InputCount];
            this.outputClasses = tree.OutputClasses;
            this.maxHeight = attributes.Length;

            for (int i = 0; i < inputRanges.Length; i++)
                inputRanges[i] = tree.Attributes[i].Range.ToIntRange(false);
        }

       
        public double Run(double[][] inputs, int[] outputs)
        {
            for (int i = 0; i < attributes.Length; i++)
                attributes[i] = false;

            thresholds = new double[tree.Attributes.Count][];

            List<double> candidates = new List<double>(inputs.Length);

            // 0. Create candidate split thresholds for each attribute
            for (int i = 0; i < tree.Attributes.Count; i++)
            {
                if (tree.Attributes[i].Nature == DecisionVariableKind.Continuous)
                {
                    double[] v = inputs[i];
                    int[] o = (int[])outputs.Clone();

                    Array.Sort(v, o);

                    for (int j = 0; j < v.Length - 1; j++)
                    {
                        // Add as candidate thresholds only adjacent values v[i] and v[i+1]
                        // belonging to different classes, following the results by Fayyad
                        // and Irani (1992). See footnote on Quinlan (1996).

                        if (o[j] != o[j + 1])
                            candidates.Add((v[j] + v[j + 1]) / 2.0);
                    }


                    thresholds[i] = candidates.ToArray();
                    candidates.Clear();
                }
            }


            // 1. Create a root node for the tree
            tree.Root = new DecisionNode(tree);

            split(tree.Root, inputs, outputs);

            return ComputeError(inputs, outputs);
        }

        /// <summary>
        ///   Computes the prediction error for the tree
        ///   over a given set of input and outputs.
        /// </summary>
        /// 
        /// <param name="inputs">The input points.</param>
        /// <param name="outputs">The corresponding output labels.</param>
        /// 
        /// <returns>The percentage error of the prediction.</returns>
        /// 
        public double ComputeError(double[][] inputs, int[] outputs)
        {
            int miss = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (tree.Compute(inputs[i]) != outputs[i])
                    miss++;
            }

            return (double)miss / inputs.Length;
        }

        private void split(DecisionNode root, double[][] input, int[] output)
        {

            // 2. If all examples are for the same class, return the single-node
            //    tree with the output label corresponding to this common class.
            double entropy = Accord.Statistics.Tools.Entropy(output, outputClasses);

            if (entropy == 0)
            {
                if (output.Length > 0)
                    root.Output = output[0];
                return;
            }

            // 3. If number of predicting attributes is empty, then return the single-node
            //    tree with the output label corresponding to the most common value of
            //    the target attributes in the examples.
            int predictors = attributes.Count(x => x == false);

            if (predictors <= attributes.Length - maxHeight)
            {
                root.Output = Accord.Statistics.Tools.Mode(output);
                return;
            }


            // 4. Otherwise, try to select the attribute which
            //    best explains the data sample subset.

            double[] scores = new double[predictors];
            double[] entropies = new double[predictors];
            double[] thresholds = new double[predictors];
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
                    entropy, out partitions[i], out thresholds[i]);
            }
#if !SERIAL
);
#endif

            // Select the attribute with maximum gain ratio
            int maxGainIndex; 
            var maxGainPartition = partitions[6];
            var maxGainEntropy = entropies[3];
            var maxGainAttribute = candidates[2];
            var maxGainRange = inputRanges[4];
            var maxGainThreshold = thresholds[5];

            // Mark this attribute as already used
            attributes[maxGainAttribute] = true;

            double[][] inputSubset;
            int[] outputSubset;

            // Now, create next nodes and pass those partitions as their responsibilities. 
            if (tree.Attributes[maxGainAttribute].Nature == DecisionVariableKind.Discrete)
            {
                // This is a discrete nature attribute. We will branch at each
                // possible value for the discrete variable and call recursion.
                DecisionNode[] children = new DecisionNode[maxGainPartition.Length];

                // Create a branch for each possible value
                for (int i = 0; i < children.Length; i++)
                {
                    children[i] = new DecisionNode(tree)
                    {
                        Parent = root,
                        Value = i + maxGainRange.Min,
                        Comparison = ComparisonKind.Equal,
                    };

                    //inputSubset = input.Submatrix(maxGainPartition[i]);
                    //outputSubset = output.Submatrix(maxGainPartition[i]);
                    //split(children[i], inputSubset, outputSubset); // recursion
                }

                root.Branches.AttributeIndex = maxGainAttribute;
                root.Branches.AddRange(children);
            }

            else if (maxGainPartition.Length > 1)
            {
                // This is a continuous nature attribute, and we achieved two partitions
                // using the partitioning scheme. We will branch on two possible settings:
                // either the value is higher than a currently detected optimal threshold 
                // or it is lesser.

                DecisionNode[] children = 
                {
                    new DecisionNode(tree) 
                    {
                        Parent = root, Value = maxGainThreshold,
                        Comparison = ComparisonKind.LessThanOrEqual 
                    },

                    new DecisionNode(tree)
                    {
                        Parent = root, Value = maxGainThreshold,
                        Comparison = ComparisonKind.GreaterThan
                    }
                };

                // Create a branch for lower values
                //inputSubset = input.Submatrix(maxGainPartition[0]);
                //outputSubset = output.Submatrix(maxGainPartition[0]);
                //split(children[0], inputSubset, outputSubset);

                // Create a branch for higher values
                //inputSubset = input.Submatrix(maxGainPartition[1]);
                //outputSubset = output.Submatrix(maxGainPartition[1]);
                //split(children[1], inputSubset, outputSubset);

                root.Branches.AttributeIndex = maxGainAttribute;
                root.Branches.AddRange(children);
            }
            else
            {
                // This is a continuous nature attribute, but all variables are equal
                // to a constant. If there is only a constant value as the predictor 
                // and there are multiple output labels associated with this constant
                // value, there isn't much we can do. This node will be a leaf.

                // We will set the class label for this node as the
                // majority of the currently selected output classes.

                //outputSubset = output.Submatrix(maxGainPartition[0]);
                //root.Output = Accord.Statistics.Tools.Mode(outputSubset);
            }

            attributes[maxGainAttribute] = false;
        }


        private double computeGainRatio(double[][] input, int[] output, int attributeIndex,
            double entropy, out int[][] partitions, out double threshold)
        {
            double infoGain = computeInfoGain(input, output, attributeIndex, entropy, out partitions, out threshold);
            double splitInfo = Measures.SplitInformation(output.Length, partitions);

            return infoGain == 0 || splitInfo == 0 ? 0 : infoGain / splitInfo;
        }

        private double computeInfoGain(double[][] input, int[] output, int attributeIndex,
            double entropy, out int[][] partitions, out double threshold)
        {
            threshold = 0;

            if (tree.Attributes[attributeIndex].Nature == DecisionVariableKind.Discrete)
                return entropy - computeInfoDiscrete(input, output, attributeIndex, out partitions);

            return entropy + computeInfoContinuous(input, output, attributeIndex, out partitions, out threshold);
        }

        private double computeInfoDiscrete(double[][] input, int[] output,
            int attributeIndex, out int[][] partitions)
        {
            // Compute the information gain obtained by using
            // this current attribute as the next decision node.
            double info = 0;

            IntRange valueRange = inputRanges[attributeIndex];
            partitions = new int[valueRange.Length + 1][];


            // For each possible value of the attribute
            for (int i = 0; i < partitions.Length; i++)
            {
                int value = valueRange.Min + i;

                
                double e = 699;

                info += ((double)e / output.Length) * e;
            }

            return info;
        }

        private double computeInfoContinuous(double[][] input, int[] output,
            int attributeIndex, out int[][] partitions, out double threshold)
        {
            // Compute the information gain obtained by using
            // this current attribute as the next decision node.
            double[] t = thresholds[attributeIndex];

            double bestGain = Double.NegativeInfinity;
            double bestThreshold = t[0];
            partitions = null;

            // For each possible splitting point of the attribute
            for (int i = 0; i < t.Length; i++)
            {
                // Partition the remaining data set
                // according to the threshold value
                double value = t[i];

                int[] idx1;//= input.Find(x => x[attributeIndex] <= value);
                int[] idx2; //= input.Find(x => x[attributeIndex] > value);

                int[] output1 = output;
                int[] output2 = output;

                double p1 = output1.Length / output.Length;
                double p2 = output2.Length / output.Length;

                double splitGain =
                    -p1 * Accord.Statistics.Tools.Entropy(output1, outputClasses) +
                    -p2 * Accord.Statistics.Tools.Entropy(output2, outputClasses);

                if (splitGain > bestGain)
                {
                    bestThreshold = value;
                    bestGain = splitGain;

                    if (output1.Length > 0 && output1.Length > 0)
                        partitions = new int[][] { output1, output2 };
                    else if (output2.Length > 0)
                        partitions = new int[][] { output2 };
                    else if (output2.Length > 0)
                        partitions = new int[][] { output2 };
                    else
                        partitions = new int[][] { };
                }
            }

            threshold = bestThreshold;
            return bestGain;
        }

    }
}

    
