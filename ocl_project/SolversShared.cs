#define HOST_PARALLEL

using System;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;

using Real = double;
using VectorReal = double[];
using VectorInt = int[];
using System.Collections.Concurrent;

namespace Solvers
{
    public static class Shared
    {
        public const int MAX_ITER = (int)1e+3;
        public const Real EPS = 1e-13F;
    
        public static void MSRMul(
            VectorReal mat,
            VectorInt aptr,
            VectorInt jptr,
            int n,
            VectorReal v,
            VectorReal res)
        {
            MyFor(0, n, i => {
                int start = aptr[i];
                int stop = aptr[i + 1];
                Real dot = mat[i] * v[i];
                for (int a = start; a < stop; a++)
                {
                    dot += mat[a] * v[jptr[a - n]];
                }
                res[i] = dot;
            });
        }
        
        public static Real Dot(
            VectorReal x,
            VectorReal y)
        {
            Real acc = 0;

            for (int i = 0; i < x.Length; i++)
            {
                acc += x[i] * y[i];
            }

            return acc;
        }

        public static void MyFor(int i0, int i1, Action<int> iteration)
        {
#if HOST_PARALLEL
            var partitioner = Partitioner.Create(i0, i1);
            Parallel.ForEach(partitioner, (range, state) =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                {
                    iteration(i);
                }
            });
#else
            for (int i = i0; i < i1; i++)
            {
                iteration(i);
            }        
#endif
        }
        
        public static T[] LoadArray<T>(StreamReader file) where T : INumber<T>
        {
            var sizeStr = file.ReadLine();
            var size = int.Parse(sizeStr!);
            var array = new T[size];

            for (int i = 0; i < (int)size; i++)
            {
                var row = file.ReadLine();
                T elem;
                try
                {
                    elem = T.Parse(row!, CultureInfo.InvariantCulture);
                }
                catch (SystemException)
                {
                    throw new System.Exception($"i = {i}");
                }

                try
                {
                    array[i]=elem;
                }
                catch (SystemException)
                {
                    throw new System.Exception($"Out of Range: i = {i}, size = {size}");
                }
            }

            return array;
        }
        
        public static void PrintArray<T>(SparkCL.Memory<T> mem, int count = 5)
        where T: unmanaged, INumber<T> 
        {
            for (int i = 0; i < count; i++)
            {
                Console.WriteLine(mem[i]);
            }
            
            Console.WriteLine("- - -");
            
            for (int i = (int)mem.Count - count; i < (int)mem.Count; i++)
            {
                Console.WriteLine(mem[i]);
            }
        }
        
        public static void PrintArray<T>(T[] mem, int count = 5)
        where T: unmanaged, INumber<T> 
        {
            for (int i = 0; i < count; i++)
            {
                Console.WriteLine(mem[i]);
            }
            
            Console.WriteLine("- - -");
            
            for (int i = mem.Length - count; i < mem.Length; i++)
            {
                Console.WriteLine(mem[i]);
            }
        }
    }
    
    public class HostSlae
    {
        public VectorReal mat ;
        public VectorReal f   ;
        public VectorInt  aptr;
        public VectorInt  jptr;
        public VectorReal x   ;
        public VectorReal ans ;

        public HostSlae()
        {
            mat  = Shared.LoadArray<Real>(File.OpenText("../../../../slae/mat"));
            f    = Shared.LoadArray<Real>(File.OpenText("../../../../slae/f"));
            aptr = Shared.LoadArray<int> (File.OpenText("../../../../slae/aptr"));
            jptr = Shared.LoadArray<int> (File.OpenText("../../../../slae/jptr"));
            x    = Shared.LoadArray<Real>(File.OpenText("../../../../slae/x"));
            ans  = Shared.LoadArray<Real>(File.OpenText("../../../../slae/ans"));
        }
    }
    
    public class OclSlae
    {
        public SparkCL.Memory<Real> mat ;
        public SparkCL.Memory<Real> f   ;
        public SparkCL.Memory<int>  aptr;
        public SparkCL.Memory<int>  jptr;
        public SparkCL.Memory<Real> x   ;
        public SparkCL.Memory<Real> ans ;

        public OclSlae()
        {
            mat  = new SparkCL.Memory<Real>(File.OpenText("../../../../slae/mat"));
            f    = new SparkCL.Memory<Real>(File.OpenText("../../../../slae/f"));
            aptr = new SparkCL.Memory<int> (File.OpenText("../../../../slae/aptr"));
            jptr = new SparkCL.Memory<int> (File.OpenText("../../../../slae/jptr"));
            x    = new SparkCL.Memory<Real>(File.OpenText("../../../../slae/x"));
            ans  = new SparkCL.Memory<Real>(File.OpenText("../../../../slae/ans"));
        }
    }
}
