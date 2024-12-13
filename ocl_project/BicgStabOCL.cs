using System;
using System.Diagnostics;

using static Solvers.Shared;

using Real = float;

namespace Solvers.OpenCL
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
                kernMul.SetArg(0, mat);
                kernMul.SetArg(1, aptr);
                kernMul.SetArg(2, jptr);
                kernMul.SetArg(3, (uint)x.Count);
            
            void MulExecute(SparkCL.Memory<Real> a, SparkCL.Memory<Real> b){
                kernMul.SetArg(4, a);
                kernMul.SetArg(5, b);
                kernMul.Execute();
            }
                
            var kernAxpy = solvers.GetKernel(
                "BLAS_axpy",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
            void AxpyExecute(Real _a, SparkCL.Memory<Real> _x, SparkCL.Memory<Real> _y) {
                kernAxpy.SetArg(0, _a);
                kernAxpy.SetArg(1, _x);
                kernAxpy.SetArg(2, _y);
                kernAxpy.Execute();
            }
            
            var kernScale = solvers.GetKernel(
                "BLAS_scale",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
            void ScaleExecute(Real _a, SparkCL.Memory<Real> _x) {
                kernAxpy.SetArg(0, _a);
                kernAxpy.SetArg(1, _x);
                kernAxpy.Execute();
            }

            // BiCGSTAB
            prepare1.Execute();
            r.Read();
            
            r.CopyTo(r_hat);
            r.CopyTo(p);
            
            r_hat.Read();
            p.Read();

            int iter = 0;
            Real rr = 0;

            sw_host.Start();
            Real pp = r.Dot(r); // r_hat * r
            sw_host.Stop();

            for (; iter < MAX_ITER; iter++)
            {
                MulExecute(p, nu);
                nu.Read();

                sw_host.Start();
                Real rnu = r_hat.Dot(nu);
                Real alpha = pp / rnu;
                sw_host.Stop();

                // 3. h = x + alpha*p
                x.CopyTo(h);
                AxpyExecute(alpha, p, h);
                // 4.
                r.CopyTo(s);
                AxpyExecute(-alpha, nu, s);

                s.Read();
                h.Read();

                sw_host.Start();
                Real ss = s.Dot(s);
                if (ss < EPS)
                {
                    x.Dispose();
                    x = h;
                    break;
                }
                sw_host.Stop();

                MulExecute(s, t);
                t.Read();

                sw_host.Start();
                Real ts = t.Dot(s);
                Real tt = t.Dot(t);
                Real w = ts / tt;
                sw_host.Stop();

                // 8. 
                h.CopyTo(x);
                AxpyExecute(w, s, x);

                // 9.
                s.CopyTo(r);
                AxpyExecute(-w, t, r);                

                r.Read();
                x.Read();

                sw_host.Start();
                rr = r.Dot(r);
                if (rr < EPS)
                {
                    break;
                }

                // 11-12.
                Real pp1 = r_hat.Dot(r);
                Real beta = (pp1 / pp) * (alpha / w);
                sw_host.Stop();

                // 13.
                AxpyExecute(-w, nu, p);
                p.Read();

                // что если объединить два последних действия в одно ядро?
                ScaleExecute(beta, p);
                p.Read();

                AxpyExecute(1f, r, p);
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
