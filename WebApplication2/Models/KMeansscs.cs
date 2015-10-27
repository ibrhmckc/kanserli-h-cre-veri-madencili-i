using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System;
    using Accord.Math;
    using Accord.Statistics.Distributions.Univariate;
using Accord.MachineLearning;
namespace WebApplication2.Models
{
    public class KMeansscs
    {

  
    public class KMeans : IClusteringAlgorithm<double[]>
    {

        private KMeansClusterCollection clusters;


       
        public KMeansClusterCollection Clusters
        {
            get { return clusters; }
        }

     
        public int K
        {
            get { return clusters.Count; }
        }

     
        public int Dimension
        {
            get { return clusters.Centroids[0].Length; }
        }

      
        public Func<double[], double[], double> Distance
        {
            get { return clusters.Distance; }
            set { clusters.Distance = value; }
        }

        public KMeans(int k)
            : this(k, Accord.Math.Distance.SquareEuclidean) { }

      
        public KMeans(int k, Func<double[], double[], double> distance)
        {
            if (k <= 0) 
                throw new ArgumentOutOfRangeException("k");
            if (distance == null)
                throw new ArgumentNullException("distance");

         
            this.clusters = new KMeansClusterCollection(k, distance);
        }


    
        public void Randomize(double[][] points, bool useSeeding = true)
        {
            if (points == null) 
                throw new ArgumentNullException("points");

            double[][] centroids = clusters.Centroids;

            if (useSeeding)
            {
   
                centroids[0] = (double[])points[Accord.Math.Tools.Random.Next(0, points.Length)].Clone();

                for (int c = 1; c < centroids.Length; c++)
                {
                 
                    double sum = 0;
                    double[] D = new double[points.Length];
                    for (int i = 0; i < D.Length; i++)
                    {
                        double[] x = points[i];

                        double min = Distance(x, centroids[0]);
                        for (int j = 1; j < c; j++)
                        {
                            double d = Distance(x, centroids[j]);
                            if (d < min) min = d;
                        }

                        D[i] = min;
                        sum += min;
                    }

                    for (int i = 0; i < D.Length; i++)
                        D[i] /= sum;

                 
                    centroids[c] = (double[])points[GeneralDiscreteDistribution.Random(D)].Clone();
                }
            }
            else
            {
                // pick K unique random indexes in the range 0..n-1
                int[] idx = Accord.Statistics.Tools.RandomSample(points.Length, K);

                // assign centroids from data set
                centroids = points.Submatrix(idx).MemberwiseClone();
            }

            this.clusters.Centroids = centroids;
        }

 
        public int[] Compute(double[][] points, double threshold)
        {
            return Compute(points, threshold, true);
        }

        
        public int[] Compute(double[][] data, double threshold = 1e-5, bool computeInformation = true)
        {
            // Initial argument checking
            if (data == null)
                throw new ArgumentNullException("data");
            if (data.Length < K)
                throw new ArgumentException("Not enough points. There should be more points than the number K of clusters.");
            if (threshold < 0)
                throw new ArgumentException("Threshold should be a positive number.", "threshold");

          

            int k = this.K;
            int rows = data.Length;
            int cols = data[0].Length;


            // Perform a random initialization of the clusters
            // if the algorithm has not been initialized before.
            if (this.Clusters.Centroids[0] == null)
            {
                Randomize(data, useSeeding: false);
            }


            // Initial variables
            int[] count = new int[k];
            int[] labels = new int[rows];
            double[][] centroids = clusters.Centroids;
            double[][] newCentroids = new double[k][];
            for (int i = 0; i < newCentroids.Length; i++)
                newCentroids[i] = new double[cols];

            double[][,] covariances = clusters.Covariances;
            double[] proportions = clusters.Proportions;


            bool shouldStop = false;

            while (!shouldStop) // Main loop
            {
                // Reset the centroids and the member counters
                for (int i = 0; i < newCentroids.Length; i++)
                    Array.Clear(newCentroids[i], 0, newCentroids[i].Length);
                Array.Clear(count, 0, count.Length);

             
                for (int i = 0; i < data.Length; i++)
                {
                    // Get the point
                    double[] point = data[i];

                    // Get the nearest cluster centroid
                    int c = labels[i] = Clusters.Nearest(point);

                    // Increase the cluster's sample counter
                    count[c]++;

                    // Accumulate in the corresponding centroid
                    for (int j = 0; j < point.Length; j++)
                        newCentroids[c][j] += point[j];
                }

                for (int i = 0; i < newCentroids.Length; i++)
                {
                    double clusterCount = count[i];

                    if (clusterCount != 0)
                    {
                        for (int j = 0; j < newCentroids[i].Length; j++)
                            newCentroids[i][j] /= clusterCount;
                    }
                }


              
                shouldStop = converged(centroids, newCentroids, threshold);

                // go to next generation
                for (int i = 0; i < centroids.Length; i++)
                    for (int j = 0; j < centroids[i].Length; j++)
                        centroids[i][j] = newCentroids[i][j];
            }


            if (computeInformation)
            {
                // Compute cluster information (optional)
                for (int i = 0; i < centroids.Length; i++)
                {
                    // Extract the data for the current cluster
                    double[][] sub = data.Submatrix(labels.Find(x => x == i));

                    if (sub.Length > 0)
                    {
                        // Compute the current cluster variance
                        covariances[i] = Accord.Statistics.Tools.Covariance(sub, centroids[i]);
                    }
                    else
                    {
                        // The cluster doesn't have any samples
                        covariances[i] = new double[cols, cols];
                    }

                    // Compute the proportion of samples in the cluster
                    proportions[i] = (double)sub.Length / data.Length;
                }
            }

            clusters.Centroids = centroids;

            // Return the classification result
            return labels;
        }

 
        public int[] Compute(double[][] data, out double error, bool computeInformation = true)
        {
            return Compute(data, 1e-5, out error, computeInformation);
        }

   
        public int[] Compute(double[][] data, double threshold, out double error, bool computeInformation = true)
        {
            // Initial argument checking
            if (data == null) throw new ArgumentNullException("data");

            // Classify the input data
            int[] labels = Compute(data, threshold, computeInformation);

            // Compute the average error
            error = Clusters.Distortion(data, labels);

            // Return the classification result
            return labels;
        }

       
        private static bool converged(double[][] centroids, double[][] newCentroids, double threshold)
        {
            for (int i = 0; i < centroids.Length; i++)
            {
                double[] centroid = centroids[i];
                double[] newCentroid = newCentroids[i];

                for (int j = 0; j < centroid.Length; j++)
                {
                    if ((System.Math.Abs((centroid[j] - newCentroid[j]) / centroid[j])) >= threshold)
                        return false;
                }
            }
            return true;
        }

     
        IClusterCollection<double[]> IClusteringAlgorithm<double[]>.Clusters
        {
            get { return clusters; }
        }



        #region Deprecated
         
        [Obsolete("Please use Clusters.Nearest() instead.")]
        public int Nearest(double[] point)
        {
            return Clusters.Nearest(point);
        }

       
        [Obsolete("Please use Clusters.Nearest() instead.")]
        public int[] Nearest(double[][] points)
        {
            return Clusters.Nearest(points);
        }
        #endregion

    }



}

    }
