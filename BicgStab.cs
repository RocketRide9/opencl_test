//#define HOST_PARALLEL
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
using VectorReal = System.Collections.Generic.List<double>;
using VectorInt = System.Collections.Generic.List<int>;
namespace CPU_TEST
{
    public static class CPU
    {
        const int MAX_ITER = (int)1e+3;
        const Real EPS = 1e-13F;



        static List<T> LoadArray<T>(StreamReader file) where T : INumber<T>
        {
            var sizeStr = file.ReadLine();
            var size = int.Parse(sizeStr!);
            var array = new List<T>(size);

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
                    array.Add(elem);
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
            public VectorReal mat =new();
            public VectorReal f =new ();
            public VectorInt aptr = new() ;
            public VectorInt jptr = new() ;
            public VectorReal x = new()  ;
            public VectorReal ans = new() ;

            public void LoadFromFiles()
            {
                 mat = LoadArray<Real>(File.OpenText("./mat"));
                 f = LoadArray<Real>(File.OpenText("./f"));
                 aptr = LoadArray<int>(File.OpenText("./aptr"));
                 jptr = LoadArray<int>(File.OpenText("./jptr"));
                 x = LoadArray<Real>(File.OpenText("./x"));
                 ans = LoadArray<Real>(File.OpenText("./ans"));
            }
            public (Real rr, Real pp, VectorReal x, int iter) Solve()
            {
                var r = new List<Real>(new Real[x.Count]);
                var r_hat = new List<Real>(new Real[x.Count]);
                var p = new List<Real>(new Real[x.Count]);
                var nu = new List<Real>(new Real[x.Count]);
                var h = new List<Real>(new Real[x.Count]);
                var s = new List<Real>(new Real[x.Count]);
                var t = new List<Real>(new Real[x.Count]);

                // var f32 = new SparkCL.Memory<Real>(1);

                // BiCGSTAB

                MSRMul(mat, aptr, jptr, x.Count, x, ref t);
                MyFor(0, x.Count, i =>
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
                    MSRMul(mat, aptr, jptr, x.Count, p, ref nu);

                    Real rnu = Dot(nu, r_hat);
                    Real alpha = pp / rnu;

                    MyFor(0, x.Count, i =>
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

                    MSRMul(mat, aptr, jptr, x.Count, s, ref t);

                    Real ts = Dot(s, t);
                    Real tt = Dot(t, t);
                    Real w = ts / tt;

                    MyFor(0, x.Count, i =>
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

                    MyFor(0, x.Count, i =>
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
            for (int i = 0; i < (int)x.Count; i++)
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
            List<Real> mat,
            List<int> aptr,
            List<int> jptr,
            int n,
            List<Real> v,
            ref List<Real> res)
        {
            // var temp = new List<Real>(new Real[res.Count]);

            /*
            Parallel.For(0, n, (i, status) =>
            {
                int start = aptr[i];
                int stop = aptr[i + 1];
                Real dot = mat[i] * v[i];
                for (int a = start; a < stop; a++)
                {
                    dot += mat[a] * v[jptr[a - n]];
                }
                temp[i] = dot;
            });
            */

            var binding = res;
            MyFor(0, n, i =>
            {
                int start = aptr[i];
                int stop = aptr[i + 1];
                Real dot = mat[i] * v[i];
                for (int a = start; a < stop; a++)
                {
                    dot += mat[a] * v[jptr[a - n]];
                }
                binding[i] = dot;
            });

            // res = temp;
        }

        public static Real Dot(
            VectorReal x,
            VectorReal y)
        {
            Real acc = 0;

            for (int i = 0; i < x.Count; i++)
            {
                acc += x[i] * y[i];
            }

            return acc;
        }

        static void MyFor(int i0, int i1, Action<int> iteration)
        {
#if HOST_PARALLEL
                Parallel.For(i0, i1, (i) =>
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
    }
}
