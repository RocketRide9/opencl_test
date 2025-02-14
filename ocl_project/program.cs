using Silk.NET.OpenCL;
using System;
using System.Diagnostics;
using System.IO;
using Solvers;

using Real = float;
using System.Linq;
// using SparkOCL;
// using SparkCL;

namespace HelloWorld
{
    internal class Program
    {
        const int MAX_ITER = (int)1e+3;
        const Real EPS = 1e-13F;
        const bool HOST_PARALLEL = true;

        static void Main(string[] args)
        {
            SparkCL.Core.Init();

            // axpy_test();
            // return;

             var s1 = new Solvers.Host.BiCGStab();
            s1.SolveAndBreakdown();

            Console.WriteLine();

            var s2 = new Solvers.MKL.BiCGStab();
            s2.SolveAndBreakdown();
            
            Console.WriteLine();
            
            using var s3 = new Solvers.OpenCL.BiCGStab();
            s3.SolveAndBreakdown();
        }
        
        /*
        static void axpy_test()
        {
            var arr = Enumerable.Range(1, 256).Select(i => (Real)0.5).ToArray();
            var x           = new SparkCL.Memory<Real>(arr);
            var y           = new SparkCL.Memory<Real>(arr);
            var z           = new SparkCL.Memory<Real>(arr);
            var dotpart     = new SparkCL.Memory<Real>(32*2);
            var dotres      = new SparkCL.Memory<Real>(1);

            var solvers = new SparkCL.Program("Solvers.cl");
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

                dotres.Read();
                return dotres[0];
            }

            var res = DotExecute(x, x);
            // z.Read();1
            Console.WriteLine(res);
            // for (int i = 0; i < (int)z.Count; i++)
            // {
            //     Console.WriteLine(z[i]);
            // }
        }

        */
        static void DOT()
        {
            SparkCL.Core.Init();

            using var d1 = new SparkCL.Memory<Real>(File.OpenText("./x"));
            using var d2 = new SparkCL.Memory<Real>(File.OpenText("./x"));
            using var f32 = new SparkCL.Memory<Real>([0]);

            var sw_gpu = new Stopwatch();
            sw_gpu.Start();

            d1.Write();
            d2.Write();
            var solvers = new SparkCL.Program("Solvers.cl");
            var dot = solvers.GetKernel(
                "dot_kernel",
                globalWork: new(16),
                localWork:  new(16)
            );
                dot.SetArg(2, d1.Count);
                dot.SetArg(3, f32);
                dot.SetArg(4, 16);

            dot.SetArg(0, d1);
            dot.SetArg(1, d2);
            dot.Execute();
            f32.Read();

            sw_gpu.Stop();

            Console.WriteLine($"dot = {f32[0]}");
            Console.WriteLine($"Time = {sw_gpu.ElapsedMilliseconds}");
        }
        
        static unsafe string? GetDeviceInfo(CL cl, nint device, DeviceInfo infoEnum)
        {
            cl.GetDeviceInfo(device, infoEnum, 0, null, out nuint nameSize);
            byte[] infoBytes = new byte[nameSize / (nuint)sizeof(byte)];
            fixed (void* p = infoBytes)
            {
                cl.GetDeviceInfo(device, infoEnum, nameSize, p, null);
            }
            string? result;
            switch (infoEnum)
            {
                case DeviceInfo.MaxMemAllocSize:
                    {
                        long num;
                        if (nameSize == 8)
                        {
                            num = BitConverter.ToInt64(infoBytes);
                        }
                        else
                        {
                            num = BitConverter.ToInt32(infoBytes);
                        }
                        result = (num / 1073741824.0).ToString() + " GiB";
                        break;
                    }
                case DeviceInfo.MaxWorkGroupSize:
                    {
                        long num;
                        if (nameSize == 8)
                        {
                            num = BitConverter.ToInt64(infoBytes);
                        }
                        else
                        {
                            num = BitConverter.ToInt32(infoBytes);
                        }
                        result = num.ToString();
                        break;
                    }
                default:
                    result = System.Text.Encoding.UTF8.GetString(infoBytes);
                    break;
            }
            return result;
        }
    }
}
