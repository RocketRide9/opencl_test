using System;
using System.Diagnostics;

using static Solvers.Shared;

using Real = float;
using VectorReal = float[];
using VectorInt = int[];

namespace Solvers
{
    namespace OpenCL
    {
        public class BiCGStab : IDisposable
        {
            OclSlae slae = new();

            SparkCL.Memory<Real> r;
            SparkCL.Memory<Real> r_hat;
            SparkCL.Memory<Real> p;
            SparkCL.Memory<Real> nu;
            SparkCL.Memory<Real> h;
            SparkCL.Memory<Real> s;
            SparkCL.Memory<Real> t;
            private bool disposedValue;

            public BiCGStab()
            {
                slae = new();
                slae.LoadFromFiles();
                
                r     = new SparkCL.Memory<Real>(slae.x.Count);
                r_hat = new SparkCL.Memory<Real>(slae.x.Count);
                p     = new SparkCL.Memory<Real>(slae.x.Count);
                nu    = new SparkCL.Memory<Real>(slae.x.Count);
                h     = new SparkCL.Memory<Real>(slae.x.Count);
                s     = new SparkCL.Memory<Real>(slae.x.Count);
                t     = new SparkCL.Memory<Real>(slae.x.Count);
            }
            
            public (Real, Real, int, long) Solve()
            {
                // Вынос векторов в текущую область видимости
                using var mat  = slae.mat;
                using var f    = slae.f;
                using var aptr = slae.aptr;
                using var jptr = slae.jptr;
                      var x    = slae.x;
                      var ans  = slae.ans;

                var sw_host = new Stopwatch();

                // BiCGSTAB
                var solvers = new SparkCL.Program("Solvers.cl");
    
                var prepare1 = solvers.GetKernel(
                    "BiCGSTAB_prepare1",
                    globalWork: new(x.Count),
                    localWork:  new(32)
                );
                    prepare1.PushArg(mat);
                    prepare1.PushArg(aptr);
                    prepare1.PushArg(jptr);
                    prepare1.PushArg((uint)x.Count);
                    prepare1.PushArg(r);
                    prepare1.PushArg(p);
                    prepare1.PushArg(f);
                    prepare1.PushArg(x);
    
                var kernMul = solvers.GetKernel(
                    "MSRMul",
                    globalWork: new(x.Count),
                    localWork:  new(32)
                );
                    kernMul.PushArg(mat);
                    kernMul.PushArg(aptr);
                    kernMul.PushArg(jptr);
                    kernMul.PushArg((uint)x.Count);
                    kernMul.PushArg(p);
                    kernMul.PushArg(nu);
    
                var kernXpay = solvers.GetKernel(
                    "BLAS_xpay",
                    globalWork: new(x.Count/4),
                    localWork:  new(32)
                );
    
                var kernP = solvers.GetKernel(
                    "BiCGSTAB_p",
                    globalWork: new(x.Count),
                    localWork:  new(32)
                );
                    kernP.PushArg(p);
                    kernP.PushArg(r);
                    kernP.PushArg(nu);
                    
                // BiCGSTAB
                prepare1.Execute();
                r.CopyTo(r_hat);
                r.CopyTo(p);
                r.Read();
                r_hat.Read();
                p.Read();
    
                int iter = 0;
                Real rr = 0;
    
                sw_host.Start();
                Real pp = r.Dot(r); // r_hat * r
                sw_host.Stop();
    
                for (; iter < MAX_ITER; iter++)
                {
                    kernMul.SetArg(4, p);
                    kernMul.SetArg(5, nu);
                    kernMul.Execute();
                    nu.Read();
                
                    sw_host.Start();
                    Real rnu = r_hat.Dot(nu);
                    Real alpha = pp / rnu;
                    sw_host.Stop();
                    
                    // 3. h = x + alpha*p
                    kernXpay.SetArg(0, h);
                    kernXpay.SetArg(1, x);
                    kernXpay.SetArg(2, alpha);
                    kernXpay.SetArg(3, p);
                    kernXpay.Execute();
                    // 4.
                    kernXpay.SetArg(0, s);
                    kernXpay.SetArg(1, r);
                    kernXpay.SetArg(2, -alpha);
                    kernXpay.SetArg(3, nu);
                    kernXpay.Execute();
                    
                    s.Read();
                    h.Read();
                    // print(s, 5);
                    // return;
    
                    sw_host.Start();
                    Real ss = s.Dot(s);
                    if (ss < EPS)
                    {
                        x.Dispose();
                        x = h;
                        break;
                    }
                    sw_host.Stop();
                    
                    kernMul.SetArg(4, s);
                    kernMul.SetArg(5, t);
                    kernMul.Execute();
                    t.Read();
    
                    sw_host.Start();
                    Real ts = t.Dot(s);
                    Real tt = t.Dot(t);
                    Real w = ts / tt;
                    sw_host.Stop();
    
                    // 8. 
                    kernXpay.SetArg(0, x);
                    kernXpay.SetArg(1, h);
                    kernXpay.SetArg(2, w);
                    kernXpay.SetArg(3, s);
                    kernXpay.Execute();
                    
                    // 9.
                    kernXpay.SetArg(0, r);
                    kernXpay.SetArg(1, s);
                    kernXpay.SetArg(2, -w);
                    kernXpay.SetArg(3, t);
                    kernXpay.Execute();
                    
                    r.Read();
                    x.Read();
    
                    sw_host.Start();
                    rr = r.Dot(r);
                    if (rr < EPS)
                    {
                        break;
                    }
    
                    Real pp1 = r_hat.Dot(r);
                    Real beta = (pp1 / pp) * (alpha / w);
                    sw_host.Stop();
    
                    kernP.SetArg(3, w);
                    kernP.SetArg(4, beta);
                    kernP.Execute();
                    p.Read();
                    pp = pp1;
                }
    
                return (rr, pp, iter, sw_host.ElapsedMilliseconds);
            }
            
            public void SolveAndBreakdown()
            {
                var sw_ocl = new Stopwatch();
                sw_ocl.Start();
                var (rr, pp, iter, hostTime) = Solve();
                sw_ocl.Stop();

                var x = slae.x;
                Real max_err = Math.Abs(x[0] - slae.ans[0]);
                for (int i = 0; i < (int)slae.x.Count; i++)
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
                ulong overhead = (ulong)sw_ocl.ElapsedMilliseconds
                    - (SparkCL.Core.IOTime + SparkCL.Core.KernTime) / 1_000_000 - (ulong)hostTime;
                Console.WriteLine($"Итераций: {iter}");
                Console.WriteLine($"BiCGSTAB с OpenCL: {sw_ocl.ElapsedMilliseconds}мс");
                Console.WriteLine($"Время на операции:");
                ulong ioTime = SparkCL.Core.IOTime / 1_000_000;
                if (ioTime < 1)
                {
                    Console.WriteLine($"IO: <1мс");
                } else {
                    Console.WriteLine($"IO: {ioTime}мс");
                }
                Console.WriteLine($"Код OpenCL: {SparkCL.Core.KernTime/1_000_000}мс");
                Console.WriteLine($"Вычисления на хосте: {hostTime}мс");
                Console.WriteLine($"Накладные расходы: {overhead}мс");
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!disposedValue)
                {
                    if (disposing)
                    {
                        // TODO: освободить управляемое состояние (управляемые объекты)
                    }

                    r.Dispose();
                    r_hat.Dispose();
                    p.Dispose();
                    nu.Dispose();
                    h.Dispose();
                    s.Dispose();
                    t.Dispose();
                    // TODO: освободить неуправляемые ресурсы (неуправляемые объекты) и переопределить метод завершения
                    // TODO: установить значение NULL для больших полей
                    disposedValue = true;
                }
            }

            // // TODO: переопределить метод завершения, только если "Dispose(bool disposing)" содержит код для освобождения неуправляемых ресурсов
            ~BiCGStab()
            {
                // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
                Dispose(disposing: false);
            }

            public void Dispose()
            {
                // Не изменяйте этот код. Разместите код очистки в методе "Dispose(bool disposing)".
                Dispose(disposing: true);
                GC.SuppressFinalize(this);
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
}
