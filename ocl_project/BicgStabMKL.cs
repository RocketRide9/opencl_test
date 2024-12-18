using System;
using System.Diagnostics;
using Quasar.Native;

using static Solvers.Shared;

using Real = float;
using VectorReal = float[];

namespace Solvers.MKL
{
    public class BiCGStab
    {
        HostSlae slae = new();

        VectorReal r;
        VectorReal r_hat;
        VectorReal p;
        VectorReal nu;
        VectorReal h;
        VectorReal s;
        VectorReal t;
        
        public BiCGStab()
        {
            r     = new Real[slae.x.Length];
            r_hat = new Real[slae.x.Length];
            p     = new Real[slae.x.Length];
            nu    = new Real[slae.x.Length];
            h     = new Real[slae.x.Length];
            s     = new Real[slae.x.Length];
            t     = new Real[slae.x.Length];
        }
        
        public (Real, Real, int) Solve()
        {
            // Вынос векторов в текущую область видимости
            var mat  = slae.mat;
            var f    = slae.f;
            var aptr = slae.aptr;
            var jptr = slae.jptr;
            var x    = slae.x;
            var ans  = slae.ans;
            
            // BiCGSTAB
            MSRMul(mat, aptr, jptr, x.Length, x, t);
            MyFor(0, x.Length, i =>
            {
                r[i] = f[i] - t[i];
            });
            r.CopyTo(r_hat, 0);
            r.CopyTo(p, 0);
            int iter = 0;
            Real rr = 0;
            
            Real pp = (Real)BLAS.dot(x.Length, r, r); // r_hat * r
            
            for (; iter < MAX_ITER; iter++)
            {
                MSRMul(mat, aptr, jptr, x.Length, p, nu);
                
                Real rnu = (Real)BLAS.dot(x.Length, nu, r_hat);
                Real alpha = pp / rnu;
                
                MyFor(0, x.Length, i =>
                {
                    h[i] = x[i] + alpha * p[i];
                });
                MyFor(0, x.Length, i =>
                {
                    s[i] = r[i] - alpha * nu[i];
                });

                Real ss = (Real)BLAS.dot(x.Length, s, s);
                if (ss < EPS)
                {
                    x = h;
                    break;
                }
                
                MSRMul(mat, aptr, jptr, x.Length, s, t);
                
                Real ts = (Real)BLAS.dot(x.Length, s, t);
                Real tt = (Real)BLAS.dot(x.Length, t, t);
                Real w = ts / tt;
                
                MyFor(0, x.Length, i =>
                {
                    x[i] = h[i] + w * s[i];
                });
                MyFor(0, x.Length, i =>
                {
                    r[i] = s[i] - w * t[i];
                });

                rr = (Real)BLAS.dot(x.Length, r, r);
                if (rr < EPS)
                {
                    break;
                }
                
                Real pp1 = (Real)BLAS.dot(x.Length, r, r_hat);
                Real beta = (pp1 / pp) * (alpha / w);
                
                MyFor(0, x.Length, i =>
                {
                    p[i] = r[i] + beta * (p[i] - w * nu[i]);
                });
                pp = pp1;
            }
            return (rr, pp, iter);
        }
        
        public void SolveAndBreakdown()
        {
            var sw_host = new Stopwatch();
            sw_host.Start();
            var (rr, pp, iter) = Solve();
            sw_host.Stop();

            var x = slae.x;
            Real max_err = Math.Abs(x[0] - slae.ans[0]);
            for (int i = 0; i < (int)x.Length; i++)
            {
                var err = Math.Abs(x[i] - slae.ans[i]);
                if (err > max_err)
                {
                    max_err = err;
                }
            }

            Console.WriteLine("Решение с MKL");
            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"pp = {pp}");
            Console.WriteLine($"max err. = {max_err}");
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"Вычисления на хосте: {sw_host.ElapsedMilliseconds}мс");
        }
    }
    /*
    public static T sum<T>(params IEnumerable<T> parm) where T:INumber<T>
    {
        T res=T.Zero;
        foreach (var s in parm) res += s;
        return res;
    }
    */
}
