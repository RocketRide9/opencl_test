#define HOST_PARALLEL
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using HelloWorld;
using Real = double;
using VectorReal = double[];
using VectorInt = int[];

namespace CPU_TEST
{
    public static class MKL
    {
        const int MAX_ITER = (int)1e+3;
        const Real EPS = 1e-13F;


        static T[] LoadArray<T>(StreamReader file) where T : INumber<T>
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
        public class SLAE
        {
            public VectorReal mat =[];
            public VectorReal f = [];
            public VectorInt aptr = [] ;
            public VectorInt jptr = [] ;
            public VectorReal x = []  ;
            public VectorReal ans = [] ;

            public void LoadFromFiles()
            {
                mat = LoadArray<Real>(File.OpenText("../../../test/mat"));
                f = LoadArray<Real>(File.OpenText("../../../test/f"));
                aptr = LoadArray<int>(File.OpenText("../../../test/aptr"));
                jptr = LoadArray<int>(File.OpenText("../../../test/jptr"));
                x = LoadArray<Real>(File.OpenText("../../../test/x"));
                ans = LoadArray<Real>(File.OpenText("../../../test/ans"));

            Console.WriteLine(sum(1, 2));
                Console.WriteLine(sum(1, 2,3));
                Console.WriteLine(sum(4, 5, 6,7));


            }
            public (Real rr, Real pp, VectorReal x, int iter) Solve()
            {
                var r = new Real[x.Length];
                var r_hat = new Real[x.Length];
                var p = new Real[x.Length];
                var nu = new Real[x.Length];
                var h = new Real[x.Length];
                var s = new Real[x.Length];
                var t = new Real[x.Length];

                // var f32 = new SparkCL.Memory<Real>(1);

                // BiCGSTAB

                MSRMul(mat, aptr, jptr, x.Length, x, t);
                MyFor(0, x.Length, i =>
                {
                    r[i] = f[i] - t[i];
                    r_hat[i] = r[i];
                    p[i] = r[i];

                });


                int iter = 0;
                Real rr = 0;

                Real pp = Dot(r, r); // r_hat * r

                for (; iter < MAX_ITER; iter++)
                {
                    MSRMul(mat, aptr, jptr, x.Length, p, nu);

                    Real rnu = Dot(nu, r_hat);
                    Real alpha = pp / rnu;

                    MyFor(0, x.Length, i =>
                    {
                        h[i] = x[i] + alpha * p[i];
                        s[i] = r[i] - alpha * nu[i];
                    });
                    // print(s, 5);
                    // return;

                    Real ss = Dot(s, s);
                    if (ss < EPS)
                    {
                        x = h;
                        break;
                    }

                    MSRMul(mat, aptr, jptr, x.Length, s, t);

                    Real ts = Dot(s, t);
                    Real tt = Dot(t, t);
                    Real w = ts / tt;

                    MyFor(0, x.Length, i =>
                    {
                        x[i] = h[i] + w * s[i];
                        r[i] = s[i] - w * t[i];
                    });

                    rr = Dot(r, r);
                    if (rr < EPS)
                    {
                        break;
                    }

                    Real pp1 = Dot(r, r_hat);
                    Real beta = (pp1 / pp) * (alpha / w);

                    MyFor(0, x.Length, i =>
                    {
                        p[i] = r[i] + beta * (p[i] - w * nu[i]);

                    });
                    pp = pp1;
                }
                return (rr, pp, x, iter);
            }
        }

        public static void BiCGSTAB()
        {
            // на Intel флаги не повлияли на производительность
            var slae=new SLAE();
            slae.LoadFromFiles();
            var sw_host = new Stopwatch();
            sw_host.Start();
            var (rr,pp, x,iter) = slae.Solve();
            sw_host.Stop();
            Real max_err = Math.Abs(x[0] - slae.ans[0]);
            for (int i = 0; i < (int)x.Length; i++)
            {
                var err = Math.Abs(x[i] - slae.ans[i]);
                if (err > max_err)
                {
                    max_err = err;
                }
            }

            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"pp = {pp}");
            Console.WriteLine($"max err. = {max_err}");
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"Вычисления на хосте: {sw_host.ElapsedMilliseconds}мс");
        }
        static void MSRMul(
            VectorReal mat,
            VectorInt aptr,
            VectorInt jptr,
            int n,
            VectorReal v,
            VectorReal res)
        {
            MyFor(0, n, i =>
            {
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

        static void MyFor(int i0, int i1, Action<int> iteration)
        {
#if HOST_PARALLEL
                Parallel.For(i0, i1,(i) =>
                {
                    iteration(i);
                });
#else
            {
                for (int i = i0; i < i1; i++)
                {
                    iteration(i);
                }
            }
#endif
        }
        public static T sum<T>(params IEnumerable<T> parm) where T:INumber<T>
        {
            T res=T.Zero;
            foreach (var s in parm) res += s;
            return res;
        }
    }
}
