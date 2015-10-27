using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Web.Http;
using WebApplication2.Models;

namespace WebApplication2.Controllers
{
    public class ClassificationController : ApiController
    {

        public string Sec([FromBody]Secim se)
        {
            Class1 prg = new Class1();

            prg.srv_SelectMethod(Convert.ToInt32( se.secim));
          //  prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
           // prg.srv_Calculate();


            return "OHH";

        }

        public string DecisionTree([FromBody]SampleData s)
        {
            Class1 prg = new Class1();
            prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
            prg.srv_SelectMethod(1);
            return prg.srv_Result();

        }

        public string Bayes([FromBody]SampleData s)
        {
            Class1 prg = new Class1();
            prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
            prg.srv_SelectMethod(2);
            return prg.srv_Result();

        }

        public string KNN([FromBody]SampleData s)
        {
            Class1 prg = new Class1();
            prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
            prg.srv_SelectMethod(3);
            return prg.srv_Result();
        }

        public string KMeans([FromBody]SampleData s)
        {
            Class1 prg = new Class1();
            prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
            prg.srv_SelectMethod(4);
            return prg.srv_Result();
        }

        public string CC45([FromBody]SampleData s)
        {
            Class1 prg = new Class1();
            prg.srv_GetSampleData(s.ClumpThickness, s.UniformityofCellSize, s.UniformityofCellShape, s.MarginalAdhesion, s.SingleEpithelialCellSize, s.BareNuclei, s.BlandChromatin, s.NormalNucleoli, s.Mitoses);
            prg.srv_SelectMethod(5);
            return prg.srv_Result();
        }
    }
}
