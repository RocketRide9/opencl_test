// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Silk.NET.OpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Numerics;
using System.Threading.Tasks;
using static SparkCL.EventExt;
using Real = System.Single;
// using SparkOCL;
// using SparkCL;

namespace HelloWorld
{
    internal class Program
    {
        const int MAX_ITER = (int)1e+3;
        const Real EPS = 1e-7F;

        static void Main(string[] args)
        {
            // DOT();
            //LOS_cpu();
            LOS();
        }

        static void DOT()
        {
            SparkCL.Core.Init();

            var d1 = new SparkCL.Memory<Real>(File.OpenText("./x"));
            var d2 = new SparkCL.Memory<Real>(File.OpenText("./x"));
            var f32 = new SparkCL.Memory<Real>([0]);

            var sw_gpu = new Stopwatch();
            sw_gpu.Start();

            d1.Write();
            d2.Write();
            var solvers = new SparkCL.Program("Solvers.cl");
            var dot = solvers.GetKernel("dot_kernel");
                dot.SetArg(2, d1.Count);
                dot.SetArg(3, f32);
                dot.SetArg(4, 16);

            dot.SetArg(0, d1);
            dot.SetArg(1, d2);
            dot.Execute(
                globalWork: new(16),
                localWork:  new(16)
            );
            f32.Read();

            sw_gpu.Stop();

            Console.WriteLine($"dot = {f32[0]}");
            Console.WriteLine($"Time = {sw_gpu.ElapsedMilliseconds}");
        }
        
        static void BiCGSTAB()
        {
            SparkCL.Core.Init();

            // на Intel флаги не повлияли на производительность
            var mat = new SparkCL.Memory<Real>(File.OpenText("./mat"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var f = new SparkCL.Memory<Real>(File.OpenText("./f"), MemFlags.HostNoAccess);
            var aptr = new SparkCL.Memory<int>(File.OpenText("./aptr"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var jptr = new SparkCL.Memory<int>(File.OpenText("./jptr"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var x = new SparkCL.Memory<Real>(File.OpenText("./x"), MemFlags.HostNoAccess);

            var r  = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);
            var p  = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);
            var ar = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);

            var f32 = new SparkCL.Memory<Real>(1);

            var sw_gpu = new Stopwatch();
            var sw_host = new Stopwatch();
            sw_gpu.Start();

            var solvers = new SparkCL.Program("Solvers.cl");

            var losPrepare1 = solvers.GetKernel("LOS_prepare1");
                losPrepare1.PushArg(mat);
                losPrepare1.PushArg(aptr);
                losPrepare1.PushArg(jptr);
                losPrepare1.PushArg((uint)x.Count);
                losPrepare1.PushArg(f);
                losPrepare1.PushArg(x);
                losPrepare1.PushArg(r);

            var losPrepare2 = solvers.GetKernel("LOS_prepare2");
                losPrepare2.PushArg(mat);
                losPrepare2.PushArg(aptr);
                losPrepare2.PushArg(jptr);
                losPrepare2.PushArg((uint)x.Count);
                losPrepare2.PushArg(r);
                losPrepare2.PushArg(p);

            var arMul = solvers.GetKernel("MSRMul");
                arMul.PushArg(mat);
                arMul.PushArg(aptr);
                arMul.PushArg(jptr);
                arMul.PushArg((uint)x.Count);
                arMul.PushArg(r);
                arMul.PushArg(ar);

            var losXr = solvers.GetKernel("LOS_xr");
                losXr.PushArg(f);
                losXr.PushArg(p);
                losXr.PushArg(0); // альфа
                losXr.PushArg(x);
                losXr.PushArg(r);

            var losFp = solvers.GetKernel("LOS_fp");
                losFp.PushArg(r);
                losFp.PushArg(ar);
                losFp.PushArg(0); // бета
                losFp.PushArg(f);
                losFp.PushArg(p);

            var dot = solvers.GetKernel("dot_kernel");
                dot.SetArg(2, (uint)x.Count);
                dot.SetArg(3, f32);
                dot.SetSize(4, (uint)32);

            ulong ioTime = 0;
            ulong kernelTime = 0;
            // LOS
            Real bb = f.Dot(f);
            
            var evKern = losPrepare1.Execute(
                globalWork: new(x.Count),
                localWork:  new(16));
            var evIO = r.Read();
            ioTime += evIO.GetElapsed();
            kernelTime += evKern.GetElapsed();

            evKern = losPrepare2.Execute(
                globalWork: new(x.Count),
                localWork:  new(8));
            evIO = p.Read();
            ioTime += evIO.GetElapsed();
            kernelTime += evKern.GetElapsed();


            int iter = 0;

            sw_host.Start();
            Real rr = r.Dot(r);
            sw_host.Stop();

            Real pp = 1;
            Real pp0;
            for (; iter < MAX_ITER && Math.Abs(rr/bb) > EPS; iter++)
            {
                sw_host.Start();
                pp0 = pp;
                pp = p.Dot(p);
                Real alpha = (p.Dot(r)) / pp;
                if (Real.IsNaN(alpha))
                {
                    break;
                }
                sw_host.Stop();
                rr -= alpha * alpha * pp0;
                if (rr < 0 && false)
                {
                    break;
                }

                losXr.SetArg(2, alpha);
                evKern = losXr.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = r.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

                evKern = arMul.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = ar.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

                /*
                    dot.SetArg(0, p);
                    dot.SetArg(1, ar);
                    evKern = dot.Execute(
                        globalWork: new(32),
                        localWork:  new(32)
                    );
                    evIO = f32.Read();
                    ioTime += evIO.GetElapsed();
                    kernelTime += evKern.GetElapsed();
                */
                
                sw_host.Start();
                Real par = p.Dot(ar);
                // Real par = f32[0];
                Real beta = -par / pp;
                sw_host.Stop();

                losFp.SetArg(2, beta);
                evKern = losFp.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = p.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

            }
            sw_gpu.Stop();

            Real max_err = Math.Abs(x[0] - 1);
            for (int i = 0; i < (int)x.Count; i++)
            {
                var err = x[0] - i - 1;
                if (err > max_err)
                {
                    max_err = err;
                }
            }

            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"pp = {pp}");
            Console.WriteLine($"max err. = {max_err}");
            ulong overhead = (ulong)sw_gpu.ElapsedMilliseconds
                - (ioTime + kernelTime) / 1_000_000 - (ulong)sw_host.ElapsedMilliseconds;
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"ЛОС с OpenCL: {sw_gpu.ElapsedMilliseconds}мс");
            Console.WriteLine($"Время на операции:");
            Console.WriteLine($"IO: {ioTime/1_000}мкс");
            Console.WriteLine($"Код OpenCL: {kernelTime/1_000_000}мс");
            Console.WriteLine($"Вычисления на хосте: {sw_host.ElapsedMilliseconds}мс");
            Console.WriteLine($"Накладные расходы: {overhead}мс");
        }

        static void LOS()
        {
            SparkCL.Core.Init();

            // на Intel флаги не повлияли на производительность
            var mat = new SparkCL.Memory<Real>(File.OpenText("./mat"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var f = new SparkCL.Memory<Real>(File.OpenText("./f"), MemFlags.HostNoAccess);
            var aptr = new SparkCL.Memory<int>(File.OpenText("./aptr"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var jptr = new SparkCL.Memory<int>(File.OpenText("./jptr"), MemFlags.HostNoAccess | MemFlags.ReadOnly);
            var x = new SparkCL.Memory<Real>(File.OpenText("./x"), MemFlags.HostNoAccess);

            var r  = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);
            var p  = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);
            var ar = new SparkCL.Memory<Real>(x.Count, MemFlags.HostReadOnly);

            var f32 = new SparkCL.Memory<Real>(1);

            var sw_gpu = new Stopwatch();
            var sw_host = new Stopwatch();
            sw_gpu.Start();

            var solvers = new SparkCL.Program("Solvers.cl");

            var losPrepare1 = solvers.GetKernel("LOS_prepare1");
                losPrepare1.PushArg(mat);
                losPrepare1.PushArg(aptr);
                losPrepare1.PushArg(jptr);
                losPrepare1.PushArg((uint)x.Count);
                losPrepare1.PushArg(f);
                losPrepare1.PushArg(x);
                losPrepare1.PushArg(r);

            var losPrepare2 = solvers.GetKernel("LOS_prepare2");
                losPrepare2.PushArg(mat);
                losPrepare2.PushArg(aptr);
                losPrepare2.PushArg(jptr);
                losPrepare2.PushArg((uint)x.Count);
                losPrepare2.PushArg(r);
                losPrepare2.PushArg(p);

            var arMul = solvers.GetKernel("MSRMul");
                arMul.PushArg(mat);
                arMul.PushArg(aptr);
                arMul.PushArg(jptr);
                arMul.PushArg((uint)x.Count);
                arMul.PushArg(r);
                arMul.PushArg(ar);

            var losXr = solvers.GetKernel("LOS_xr");
                losXr.PushArg(f);
                losXr.PushArg(p);
                losXr.PushArg(0); // альфа
                losXr.PushArg(x);
                losXr.PushArg(r);

            var losFp = solvers.GetKernel("LOS_fp");
                losFp.PushArg(r);
                losFp.PushArg(ar);
                losFp.PushArg(0); // бета
                losFp.PushArg(f);
                losFp.PushArg(p);

            var dot = solvers.GetKernel("dot_kernel");
                dot.SetArg(2, (uint)x.Count);
                dot.SetArg(3, f32);
                dot.SetSize(4, (uint)32);

            ulong ioTime = 0;
            ulong kernelTime = 0;
            // LOS
            Real bb = f.Dot(f);
            
            var evKern = losPrepare1.Execute(
                globalWork: new(x.Count),
                localWork:  new(16));
            var evIO = r.Read();
            ioTime += evIO.GetElapsed();
            kernelTime += evKern.GetElapsed();

            evKern = losPrepare2.Execute(
                globalWork: new(x.Count),
                localWork:  new(8));
            evIO = p.Read();
            ioTime += evIO.GetElapsed();
            kernelTime += evKern.GetElapsed();


            int iter = 0;

            sw_host.Start();
            Real rr = r.Dot(r);
            sw_host.Stop();

            Real pp = 1;
            Real pp0;
            for (; iter < MAX_ITER && Math.Abs(rr/bb) > EPS; iter++)
            {
                sw_host.Start();
                pp0 = pp;
                pp = p.Dot(p);
                Real alpha = (p.Dot(r)) / pp;
                if (Real.IsNaN(alpha))
                {
                    break;
                }
                sw_host.Stop();
                rr -= alpha * alpha * pp0;
                if (rr < 0 && false)
                {
                    break;
                }

                losXr.SetArg(2, alpha);
                evKern = losXr.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = r.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

                evKern = arMul.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = ar.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

                /*
                    dot.SetArg(0, p);
                    dot.SetArg(1, ar);
                    evKern = dot.Execute(
                        globalWork: new(32),
                        localWork:  new(32)
                    );
                    evIO = f32.Read();
                    ioTime += evIO.GetElapsed();
                    kernelTime += evKern.GetElapsed();
                */
                
                sw_host.Start();
                Real par = p.Dot(ar);
                // Real par = f32[0];
                Real beta = -par / pp;
                sw_host.Stop();

                losFp.SetArg(2, beta);
                evKern = losFp.Execute(
                    globalWork: new(x.Count),
                    localWork:  new(16)
                );
                evIO = p.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

            }
            sw_gpu.Stop();

            Real max_err = Math.Abs(x[0] - 1);
            for (int i = 0; i < (int)x.Count; i++)
            {
                var err = x[0] - i - 1;
                if (err > max_err)
                {
                    max_err = err;
                }
            }

            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"pp = {pp}");
            Console.WriteLine($"max err. = {max_err}");
            ulong overhead = (ulong)sw_gpu.ElapsedMilliseconds
                - (ioTime + kernelTime) / 1_000_000 - (ulong)sw_host.ElapsedMilliseconds;
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"ЛОС с OpenCL: {sw_gpu.ElapsedMilliseconds}мс");
            Console.WriteLine($"Время на операции:");
            Console.WriteLine($"IO: {ioTime/1_000}мкс");
            Console.WriteLine($"Код OpenCL: {kernelTime/1_000_000}мс");
            Console.WriteLine($"Вычисления на хосте: {sw_host.ElapsedMilliseconds}мс");
            Console.WriteLine($"Накладные расходы: {overhead}мс");
        }

        static List<T> LoadArray<T>(
            StreamReader file)
        where T : INumber<T>
        {
            var sizeStr = file.ReadLine();
            var size = int.Parse(sizeStr!);
            var array = new List<T>(size);

            for (int i = 0; i < (int)size; i++)
            {
                var row = file.ReadLine();
                T elem;
                try {
                    elem = T.Parse(row!, CultureInfo.InvariantCulture);
                } catch (SystemException) {
                    throw new System.Exception($"i = {i}");
                }

                try {
                    array.Add(elem);
                } catch (SystemException) {
                    throw new System.Exception($"Out of Range: i = {i}, size = {size}");
                }
            }

            return array;
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

            for (int i = 0; i < n; i++)
            {
                int start = aptr[i];
                int stop = aptr[i + 1];
                Real dot = mat[i] * v[i];
                for (int a = start; a < stop; a++)
                {
                    dot += mat[a] * v[jptr[a - n]];
                }
                res[i] = dot;
            }

            // res = temp;
        }

        static Real Dot(
            List<Real> x,
            List<Real> y)
        {
            Real acc = 0;

            for (int i = 0; i < x.Count; i++)
            {
                acc += x[i] * y[i];
            }

            return acc;
        }

        static void LOS_cpu()
        {
            var mat = LoadArray<Real>(File.OpenText("./mat"));
            var f = LoadArray<Real>(File.OpenText("./f"));
            var aptr = LoadArray<int>(File.OpenText("./aptr"));
            var jptr = LoadArray<int>(File.OpenText("./jptr"));
            var x = LoadArray<Real>(File.OpenText("./x"));

            var r  = new List<Real>(new Real[x.Count]);
            var p  = new List<Real>(new Real[x.Count]);
            var ar = new List<Real>(new Real[x.Count]);

            var sw_cpu = new Stopwatch();
            sw_cpu.Start();
            // LOS

            MSRMul(mat, aptr, jptr, x.Count, x, ref r);
            for (int i = 0; i < x.Count; i++)
            {
                r[i] = f[i] - r[i];
                f[i] = r[i];
            }
            MSRMul(mat, aptr, jptr, r.Count, r, ref p);

            int iter = 0;
            Real rr = Dot(r, r);
            Real pp = 1;
            for (; iter < MAX_ITER && Math.Abs(rr) > EPS; iter++)
            {
                pp = Dot(p, p);
                Real alpha = Dot(r, p) / pp;

                for (int i = 0; i < (int)x.Count; i++)
                {
                    x[i] += (alpha * f[i]);
                    r[i] -= (alpha * p[i]);
                }

                MSRMul(mat, aptr, jptr, x.Count, r, ref ar);

                Real par = Dot(p, ar);
                Real beta = -par / pp;

                for (int i = 0; i < (int)x.Count; i++)
                {
                    f[i] = r[i] + beta * f[i];
                    p[i] = ar[i] + beta * p[i];
                }

                rr -= alpha * alpha * pp;
            }
            sw_cpu.Stop();

            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"Время CPU: {sw_cpu.ElapsedMilliseconds}мс");
        }

        /*
        static Span<real> SimpleMul(
            Context context,
            CommandQueue commandQueue,
            Device device,
            Array<real> mat,
            Array<real> vec,
            Array<real> res)
        {
            var matBuffer = new SparkOCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, mat);
            var vecBuffer = new SparkOCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, vec);
            var resBuffer = new SparkOCL.Buffer<real>(context, MemFlags.WriteOnly | MemFlags.UseHostPtr, res);

            var globalWork = new NDRange(M_ROWS);
            var localWork = new NDRange(MT);

            // Create OpenCL kernel
            var program = SparkOCL.Program.FromFilename(context, device, "HelloWorld.cl");
            var kernel = new Kernel(program, "mul_simple");
            kernel.SetArg(0, matBuffer);
            kernel.SetArg(1, vecBuffer);
            kernel.SetArg(2, resBuffer);
            kernel.SetArg(3, M_COLS);
            kernel.SetArg(4, M_ROWS);

            // Queue the kernel up for execution across the array
            commandQueue.EnqueueNDRangeKernel(
                kernel,
                new NDRange(),
                globalWork,
                localWork);

            var res_mapped = commandQueue.EnqueueMapBuffer<real>(
                resBuffer,
                true,
                MapFlags.Read,
                0,
                M_ROWS);

            return res_mapped;
        }

        static Span<real> MSRMul (
            Context context,
            CommandQueue commandQueue,
            Device device,
            Array<real> mat,
            Array<int> jptr,
            Array<int> iptr,
            Array<real> vec,
            Array<real> res)
        {
            var matBuffer  = new SparkOCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, mat);
            var vecBuffer  = new SparkOCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, vec);
            var jptrBuffer = new SparkOCL.Buffer<int>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, jptr);
            var iptrBuffer = new SparkOCL.Buffer<int>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, iptr);
            var resBuffer  = new SparkOCL.Buffer<real>(context, MemFlags.WriteOnly | MemFlags.UseHostPtr, res);

            var globalWork = new NDRange(M_ROWS);
            var localWork = new NDRange(MT);

            // Create OpenCL kernel
            var program = SparkOCL.Program.FromFilename(context, device, "SparseMul.cl");
            var kernel = new Kernel(program, "MSRMul");
            kernel.PushArg(matBuffer);
            kernel.PushArg(vecBuffer);
            kernel.PushArg(jptrBuffer);
            kernel.PushArg(iptrBuffer);
            kernel.PushArg(M_COLS);
            kernel.PushArg(resBuffer);

            // Queue the kernel up for execution across the array
            commandQueue.EnqueueNDRangeKernel(
                kernel,
                new NDRange(),
                globalWork,
                localWork);

            var res_mapped = commandQueue.EnqueueMapBuffer<real>(
                resBuffer,
                true,
                MapFlags.Read,
                0,
                6);
            Console.WriteLine("Reading back");

            return res_mapped;
        }

        */
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
