using System;
using System.Diagnostics;
using Quasar.Native;
using static Solvers.Shared;

using Real = double;

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
        SparkCL.Memory<Real> dotpart;
        SparkCL.Memory<Real> dotres;
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
            dotpart=new SparkCL.Memory<Real>(32);
            dotres= new SparkCL.Memory<Real>(1);
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

            var kernP = solvers.GetKernel(
                "BiCGSTAB_p",
                globalWork: new(x.Count),
                localWork: new(32)
            );
                kernP.SetArg(0, p);
                kernP.SetArg(1, r);
                kernP.SetArg(2, nu);

            void PExecute(Real _w, Real _beta)
            {
                kernP.SetArg(3, _w);
                kernP.SetArg(4, _beta);
                kernP.Execute();
            }

            var kernMul = solvers.GetKernel(
                "MSRMul",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                kernMul.SetArg(0, mat);
                kernMul.SetArg(1, aptr);
                kernMul.SetArg(2, jptr);
                kernMul.SetArg(3, (uint)x.Count);
            
            void MulExecute(SparkCL.Memory<Real> _a, SparkCL.Memory<Real> _b){
                kernMul.SetArg(4, _a);
                kernMul.SetArg(5, _b);
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
            
            var kern1 = solvers.GetKernel(
                "Xdot",
                globalWork: new(x.Count),
                localWork: new(32)
            );
            var kern2 = solvers.GetKernel(
                "XdotEpilogue",
                globalWork: new(32),
                localWork: new(32)
            );
            Real DotExecute(SparkCL.Memory<Real> _x, SparkCL.Memory<Real> _y)
            {
                kern1.SetArg(0, (uint)_x.Count);
                kern1.SetArg(1, _x);
                kern1.SetArg(2, _y);
                kern1.SetArg(3, dotpart);
                kern1.Execute();

                kern2.SetArg(0, dotpart);
                kern2.SetArg(1, dotres);
                kern2.Execute();

                return dotres[0];
            }
            
            /*
            var kernScale = solvers.GetKernel(
                "BLAS_scale",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
            void ScaleExecute(Real _a, SparkCL.Memory<Real> _x) {
                kernScale.SetArg(0, _a);
                kernScale.SetArg(1, _x);
                kernScale.Execute();
            }
            */
            // BiCGSTAB
            // 1.

            prepare1.Execute();
            r.Read();
            // 2.
            r.CopyTo(r_hat);
            // 3.
            sw_host.Start();
            Real pp = r.Dot(r); // r_hat * r
            sw_host.Stop();
            //pp = DotExecute(r, r);
            // 4.
            r.CopyTo(p);
            
            r_hat.Read();
            p.Read();

            int iter = 0;
            Real rr = 0;
            for (; iter < MAX_ITER; iter++)
            {
                MulExecute(p, nu);
                nu.Read();

                sw_host.Start();
                Real rnu = nu.Dot(r_hat);
                Real alpha = pp / rnu;
                sw_host.Stop();

                // 3. h = x + alpha*p
                x.CopyTo(h);
                AxpyExecute(alpha, p, h);
                
                // 4.
                r.CopyTo(s);
                AxpyExecute(-alpha, nu, s);

                s.Read();

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
                PExecute(w, beta);

                pp = pp1;
            }

            x.Read();
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
