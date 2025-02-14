using System;
using System.Diagnostics;
using SparkCL;
using SparkOCL;
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
            dotpart=new SparkCL.Memory<Real>(32*2);
            dotres= new SparkCL.Memory<Real>(1);
        }
        
        public (Real, Real, int) Solve()
        {
            // Вынос векторов в текущую область видимости
            var mat  = slae.mat;
            var f    = slae.f;
            var aptr = slae.aptr;
            var jptr = slae.jptr;
            var x    = slae.x;

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

            Event PExecute(Real _w, Real _beta)
            {
                kernP.SetArg(3, _w);
                kernP.SetArg(4, _beta);
                return kernP.Execute();
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
            
            Event MulExecute(SparkCL.Memory<Real> _a, SparkCL.Memory<Real> _res){
                kernMul.SetArg(4, _a);
                kernMul.SetArg(5, _res);
                return kernMul.Execute();
            }
                
            var kernAxpy = solvers.GetKernel(
                "BLAS_axpy",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
            Event AxpyExecute(Real _a, SparkCL.Memory<Real> _x, SparkCL.Memory<Real> _y) {
                kernAxpy.SetArg(0, _a);
                kernAxpy.SetArg(1, _x);
                kernAxpy.SetArg(2, _y);
                return kernAxpy.Execute();
            }
            
            var kern1 = solvers.GetKernel(
                "Xdot",
                globalWork: new(32*32*2),
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
                dotres.Read(true);

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
            // 2.
            r.CopyTo(r_hat);
            // 3.
            Real pp = DotExecute(r, r); // r_hat * r
            // 4.
            r.CopyTo(p);

            int iter = 0;
            Real rr = 0;
            for (; iter < MAX_ITER; iter++)
            {
                MulExecute(p, nu);

                Real rnu = DotExecute(r_hat, nu);
                Real alpha = pp / rnu;

                // 3. h = x + alpha*p
                x.CopyTo(h);
                
                AxpyExecute(alpha, p, h);
                
                // 4.
                r.CopyTo(s);
                AxpyExecute(-alpha, nu, s);

                Real ss = DotExecute(s, s);
                if (ss < EPS)
                {
                    // тогда h - решение. Предыдущий вектор x можно освободить
                    x.Dispose();
                    slae.x = h;
                    break;
                }

                MulExecute(s, t);

                Real ts = DotExecute(s, t);
                Real tt = DotExecute(t, t);
                Real w = ts / tt;

                // 8. 
                h.CopyTo(x);
                AxpyExecute(w, s, x);

                // 9.
                s.CopyTo(r);
                AxpyExecute(-w, t, r);
                
                rr = DotExecute(r, r);
                if (rr < EPS)
                {
                    break;
                }

                // 11-12.
                Real pp1 = DotExecute(r, r_hat);
                Real beta = (pp1 / pp) * (alpha / w);

                // 13.
                PExecute(w, beta);

                Core.WaitQueue();
                pp = pp1;
            }

            x.Read(true);
            return (rr, pp, iter);
        }
        
        public void SolveAndBreakdown()
        {
            var sw_ocl = new Stopwatch();
            sw_ocl.Start();
            var (rr, pp, iter) = Solve();
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
            var (IOTime, KernTime) = SparkCL.Core.MeasureTime();
            ulong overhead = (ulong)sw_ocl.ElapsedMilliseconds
                - (IOTime + KernTime) / 1_000_000;
            Console.WriteLine($"Итерации: {iter}");
            Console.WriteLine($"BiCGSTAB с OpenCL: {sw_ocl.ElapsedMilliseconds}мс");
            Console.WriteLine($"Время на операции:");
            ulong ioTime = IOTime / 1_000_000;
            if (ioTime < 1)
            {
                Console.WriteLine($"\tIO: <1мс");
            } else {
                Console.WriteLine($"\tIO: {ioTime}мс");
            }
            Console.WriteLine($"\tКод OpenCL: {KernTime/1_000_000}мс");
            Console.WriteLine($"\tНакладные расходы: {overhead}мс");
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
                dotpart.Dispose();
                dotres.Dispose();
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
