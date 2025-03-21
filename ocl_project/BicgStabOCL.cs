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
        
        static nuint PaddedTo(int initial, int multiplier)
        {
            if (initial % multiplier == 0)
            {
                return (nuint)initial;
            } else {
                return ((nuint)initial / 32 + 1 ) * 32;
            }
        }
        
        public (Real, Real, int) Solve()
        {
            var solver = new SparkAlgos.BicgStab(
                slae.mat,
                slae.di,
                slae.f,
                slae.aptr,
                slae.jptr,
                slae.x,
                MAX_ITER
            );

            return solver.Solve();
        }
        
        public void SolveAndBreakdown()
        {
            var sw_ocl = new Stopwatch();
            sw_ocl.Start();
            var (rr, pp, iter) = Solve();
            sw_ocl.Stop();

            var x = slae.x;
            Real max_err = Math.Abs(x[0] - slae.ans[0]);
            for (int i = 0; i < slae.x.Count; i++)
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
