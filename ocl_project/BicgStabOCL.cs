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

        private bool disposedValue;

        public BiCGStab()
        {
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
        
        public (SparkOCL.Array<Real>, Real, Real, int) Solve()
        {
            var solver = new SparkAlgos.BicgStab(
                slae.mat,
                slae.di,
                slae.f,
                slae.aptr,
                slae.jptr,
                slae.x,
                MAX_ITER,
                EPS
            );

            return solver.Solve();
        }
        
        public void SolveAndBreakdown()
        {
            var sw_ocl = new Stopwatch();
            sw_ocl.Start();
            var (ans, rr, pp, iter) = Solve();
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
