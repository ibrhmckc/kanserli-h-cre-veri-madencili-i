using Accord.MachineLearning.DecisionTrees.Learning;
using System;
using Accord;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using Accord.Statistics.Filters;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.Bayes;
using Accord.Math;
using Accord.MachineLearning;

namespace WebApplication2.Models
{
    public class Class1
    {

        public string[] inputlar;
        public double[][] newInputs = new double[699][];
        public string sonuc;

        public void srv_GetSampleData(string ClumpThickness,
            string UniformityofCellSize,
            string UniformityofCellShape,
            string MarginalAdhesion,
            string SingleEpithelialCellSize,
            string BareNuclei,
            string BlandChromatin,
            string NormalNucleoli,
            string Mitoses)
        {
            //inputlar..
            inputlar = new string[]{ ClumpThickness, UniformityofCellSize ,
                                    UniformityofCellShape,MarginalAdhesion,
                                    SingleEpithelialCellSize,BareNuclei,
                                    BlandChromatin,NormalNucleoli,Mitoses};

        }

        public string srv_Result() { return tempResult; }



        public static void Main()
        {
            //srv_SelectMethod();

            //Console.ReadKey();

        }

        public string tempResult { get; set; }

        public void srv_SelectMethod(int sec)
        {

            DataTable tbl;
            
            tbl = ConvertToDataTable(System.Web.Hosting.HostingEnvironment.MapPath("~/TransientStorage/") + "kod.txt", 11);

            #region
            switch (sec)
            {
                case 1:
                    {
                        tempResult = kararAgaci(tbl);
                    } break;

                case 2:
                    {
                        tempResult = bayes(tbl);
                    } break;

                case 3:
                    {
                        tempResult = knn(tbl);
                    } break;

                case 4:
                    {
                        tempResult = knn(tbl);
                    } break;

                case 5:
                    {
                        tempResult = kararAgaci(tbl);
                    } break;
            }
            #endregion
        }

        private string C45(DataTable tbl)
        {


            int classCount = 2;
            Codification codebook = new Codification(tbl);


            DecisionVariable[] attributes ={
                                          new DecisionVariable("Clump Thickness",10),
                                          new DecisionVariable("Uniformity of Cell Size",10),new DecisionVariable("Uniformity of Cell Shape",10),
                                          new DecisionVariable("Marginal Adhesion",10),new DecisionVariable("Single Epithelial Cell Size",10),
                                          new DecisionVariable("Bare Nuclei",10),new DecisionVariable("Bland Chromatin",10),
                                          new DecisionVariable("Normal Nucleoli",10),new DecisionVariable("Mitoses",10),
                                          
                                          };






            DecisionTree tree = new DecisionTree(attributes, classCount);
           // ID3Learning id3learning = new ID3Learning(tree);

            // Translate our training data into integer symbols using our codebook:
            DataTable symbols = codebook.Apply(tbl);

            double[][] inputs = symbols.ToIntArray("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses").ToDouble();
            int[] outputs = symbols.ToIntArray("Class").GetColumn(0);

            // symbols.
           // id3learning.Run(inputs, outputs);
            // Now, let's create the C4.5 algorithm
            C45Learning c45 = new C45Learning(tree);

            // and learn a decision tree. The value of
            //   the error variable below should be 0.
            // 
            double error = c45.Run(inputs, outputs);


            // To compute a decision for one of the input points,
            //   such as the 25-th example in the set, we can use
            // 
            int y = tree.Compute(inputs[5]);

            // Finally, we can also convert our tree to a native
            // function, improving efficiency considerably, with
            // 
            Func<double[], int> func = tree.ToExpression().Compile();

            // Again, to compute a new decision, we can just use
            // 
            int z = func(inputs[5]);


            int[] query = codebook.Translate(inputlar[0], inputlar[1], inputlar[2], inputlar[3],
                inputlar[4], inputlar[5], inputlar[6], inputlar[7], inputlar[8]);
            int output = tree.Compute(query);
            string answer = codebook.Translate("Class", output);
            return answer;

           // throw new NotImplementedException();
          
        }



        public string kmeans(DataTable tbl)
        {

            Codification codebook = new Codification(tbl);

            DataTable symbols = codebook.Apply(tbl);

            double[][] inputs = symbols.ToIntArray("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses").ToDouble();
            int sayac = 0;

            int[] outputs = symbols.ToIntArray("Class").GetColumn(0);



            // Declare some observations
            //double[][] observations = 
            //    {
            //     new double[] { -5, -2, -1 },
            //     new double[] { -5, -5, -6 },
            //     new double[] {  2,  1,  1 },
            //     new double[] {  1,  1,  2 },
            //     new double[] {  1,  2,  2 },
            //     new double[] {  3,  1,  2 },
            //     new double[] { 11,  5,  4 },
            //     new double[] { 15,  5,  6 },
            //     new double[] { 10,  5,  6 },
            //    };


            KMeans kmeans = new KMeans(2);



            int[] labels = kmeans.Compute(inputs);


            int c = kmeans.Clusters.Nearest(new double[] { Convert.ToInt32(inputlar[0]), Convert.ToInt32(inputlar[1]),
                        Convert.ToInt32( inputlar[2]), Convert.ToInt32( inputlar[3]), Convert.ToInt32( inputlar[4]),
                        Convert.ToInt32( inputlar[5]), Convert.ToInt32( inputlar[6]), Convert.ToInt32( inputlar[7]), Convert.ToInt32( inputlar[8]) });
            return c.ToString(); ;
        }

        public string knn(DataTable tbl)
        {
            Codification codebook = new Codification(tbl);

            DataTable symbols = codebook.Apply(tbl);

            double[][] inputs = symbols.ToIntArray("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses").ToDouble();
            int sayac = 0;

            int[] outputs = symbols.ToIntArray("Class").GetColumn(0);



            KNearestNeighbors knn = new KNearestNeighbors(k: 4, classes: 2,
    inputs: inputs, outputs: outputs);

            int answer = knn.Compute(new double[] { Convert.ToInt32(inputlar[0]), Convert.ToInt32(inputlar[1]),
                        Convert.ToInt32( inputlar[2]), Convert.ToInt32( inputlar[3]), Convert.ToInt32( inputlar[4]),
                        Convert.ToInt32( inputlar[5]), Convert.ToInt32( inputlar[6]), Convert.ToInt32( inputlar[7]), Convert.ToInt32( inputlar[8]) }); // answer will be 2.
            if (answer == 0)
                answer = 4;
            else
                answer = 2;
            
            return answer.ToString();
        }

        public string kararAgaci(DataTable tbl)
        {
            int classCount = 2;
            Codification codebook = new Codification(tbl);


            DecisionVariable[] attributes ={
                                          new DecisionVariable("Clump Thickness",10),
                                          new DecisionVariable("Uniformity of Cell Size",10),new DecisionVariable("Uniformity of Cell Shape",10),
                                          new DecisionVariable("Marginal Adhesion",10),new DecisionVariable("Single Epithelial Cell Size",10),
                                          new DecisionVariable("Bare Nuclei",10),new DecisionVariable("Bland Chromatin",10),
                                          new DecisionVariable("Normal Nucleoli",10),new DecisionVariable("Mitoses",10),
                                          
                                          };





            
            DecisionTree tree = new DecisionTree(attributes, classCount);
            ID3Learning id3learning = new ID3Learning(tree);

            // Translate our training data into integer symbols using our codebook:
            DataTable symbols = codebook.Apply(tbl);

            int[][] inputs = symbols.ToIntArray("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses");
            int[] outputs = symbols.ToIntArray("Class").GetColumn(0);

            // symbols.
            id3learning.Run(inputs, outputs);

            int[] query = codebook.Translate(inputlar[0], inputlar[1], inputlar[2], inputlar[3],
                inputlar[4], inputlar[5], inputlar[6], inputlar[7], inputlar[8]);
            int output = tree.Compute(query);
            string answer = codebook.Translate("Class", output);
            
            return answer;
        }

        private string bayes(DataTable tbl)
        {
            Codification codebook = new Codification(tbl,
         "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class");

            // Translate our training data into integer symbols using our codebook:
            DataTable symbols = codebook.Apply(tbl);
            int[][] inputs = symbols.ToIntArray("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses");
            int[] outputs = symbols.ToIntArray("Class").GetColumn(0);


            // Gather information about decision variables
            int[] symbolCounts =
            {
                codebook["Clump Thickness"].Symbols,     // 3 possible values (Sunny, overcast, rain)
                codebook["Uniformity of Cell Size"].Symbols, // 3 possible values (Hot, mild, cool)
                codebook["Uniformity of Cell Shape"].Symbols,    // 2 possible values (High, normal)
                codebook["Marginal Adhesion"].Symbols ,        // 2 possible values (Weak, strong)
                codebook["Single Epithelial Cell Size"].Symbols  ,
                codebook["Bare Nuclei"].Symbols  ,
                codebook["Bland Chromatin"].Symbols , 
                codebook["Normal Nucleoli"].Symbols , 
                codebook["Mitoses"].Symbols  
            };

            int classCount = codebook["Class"].Symbols; // 2 possible values (yes, no)

            // Create a new Naive Bayes classifiers for the two classes
            NaiveBayes target = new NaiveBayes(classCount, symbolCounts);

            // Compute the Naive Bayes model
            target.Estimate(inputs, outputs);


            // We will be computing the label for a sunny, cool, humid and windy day:
            int[] instance = codebook.Translate(inputlar[0], inputlar[1], inputlar[2], inputlar[3],
                inputlar[4], inputlar[5], inputlar[6], inputlar[7], inputlar[8]);

            // Now, we can feed this instance to our model
            int output = target.Compute(instance);

            // Finally, the result can be translated back to one of the codewords using
            string result = codebook.Translate("Class", output); // result is "No"
            return result;

        }

        //public  void outto()
        //{
        //    int sonuc = 0, topplam = 0, deg = 0;
        //    for (int i = 0; i < girenint.Length; i++)
        //    {
        //        topplam += girenint[i];
        //    }
        //    sonuc = topplam / girenint.Length;
        //    if (sonuc > 3)
        //    {
        //        deg = 4;
        //    }
        //    else
        //    {
        //        deg = 2;
        //    }
        //}     
        public DataTable ConvertToDataTable(string filePath, int numberOfColumns)
        {
            DataTable keno = new DataTable("keno");

            keno.Columns.Add("Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class");


            string[] lines = System.IO.File.ReadAllLines(filePath);

            foreach (string line in lines)
            {
                var cols = line.Split(',');

                DataRow dr = keno.NewRow();
                for (int cIndex = 0; cIndex < cols.Length; cIndex++)
                {
                    dr[cIndex] = cols[cIndex];
                }

                keno.Rows.Add(dr);
            }

            return keno;
        }
    }

}
