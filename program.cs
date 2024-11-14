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
using Real = System.Double;
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
            // DOT();
            // LOS_cpu();
            // LOS();
            BiCGSTAB_cpu();
            Console.WriteLine();
            BiCGSTAB();
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
        
        static void BiCGSTAB()
        {
            SparkCL.Core.Init();

            // на Intel флаги не повлияли на производительность
            var mat =   new SparkCL.Memory<Real>(File.OpenText("./mat"));
            var f =     new SparkCL.Memory<Real>(File.OpenText("./f"));
            var aptr =  new SparkCL.Memory<int> (File.OpenText("./aptr"));
            var jptr =  new SparkCL.Memory<int> (File.OpenText("./jptr"));
            var x =     new SparkCL.Memory<Real>(File.OpenText("./x"));
            var ans =   LoadArray<Real>         (File.OpenText("./ans"));

            var r =     new SparkCL.Memory<Real>(x.Count);
            var r_hat = new SparkCL.Memory<Real>(x.Count);
            var p =     new SparkCL.Memory<Real>(x.Count);
            var nu =    new SparkCL.Memory<Real>(x.Count);
            var h =     new SparkCL.Memory<Real>(x.Count);
            var s =     new SparkCL.Memory<Real>(x.Count);
            var t =     new SparkCL.Memory<Real>(x.Count);

            // var f32 = new SparkCL.Memory<Real>(1);

            var sw_gpu = new Stopwatch();
            var sw_total = new Stopwatch();
            sw_gpu.Start();

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
                prepare1.PushArg(r_hat);
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

            var kernHS = solvers.GetKernel(
                "BiCGSTAB_hs",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                kernHS.PushArg(h);
                kernHS.PushArg(s);
                kernHS.PushArg(p);
                kernHS.PushArg(nu);
                kernHS.PushArg(x);
                kernHS.PushArg(r);
                
            var kernXR = solvers.GetKernel(
                "BiCGSTAB_xr",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                kernXR.PushArg(x);
                kernXR.PushArg(r);
                kernXR.PushArg(h);
                kernXR.PushArg(s);
                kernXR.PushArg(t);

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
            r.Read();
            r_hat.Read();
            p.Read();


            int iter = 0;
            Real rr = 0;

            sw_total.Start();
            Real pp = r.Dot(r); // r_hat * r
            sw_total.Stop();

            for (; iter < MAX_ITER; iter++)
            {
                kernMul.SetArg(4, p);
                kernMul.SetArg(5, nu);
                kernMul.Execute();
                nu.Read();
            
                sw_total.Start();
                Real rnu = r_hat.Dot(nu);
                Real alpha = pp / rnu;
                if (Real.IsNaN(alpha))
                {
                    Console.WriteLine("альфа говно");
                    break;
                }
                sw_total.Stop();

                x.Write();
                p.Write();
                r.Write();
                kernHS.SetArg(6, alpha);
                kernHS.Execute();
                s.Read();
                h.Read();
                // print(s, 5);
                // return;

                sw_total.Start();
                Real ss = s.Dot(s);
                if (ss < EPS)
                {
                    x = h;
                    break;
                }
                sw_total.Stop();
                
                kernMul.SetArg(4, s);
                kernMul.SetArg(5, t);
                kernMul.Execute();
                t.Read();

                sw_total.Start();
                Real ts = t.Dot(s);
                Real tt = t.Dot(t);
                Real w = ts / tt;
                sw_total.Stop();

                kernXR.SetArg(5, w);
                kernXR.Execute();
                r.Read();
                x.Read();

                sw_total.Start();
                rr = r.Dot(r);
                if (rr < EPS)
                {
                    break;
                }

                Real pp1 = r_hat.Dot(r);
                Real beta = (pp1 / pp) * (alpha / w);
                sw_total.Stop();

                kernP.SetArg(3, w);
                kernP.SetArg(4, beta);
                kernP.Execute();
                p.Read();
                pp = pp1;
            }
            sw_gpu.Stop();

            Real max_err = Math.Abs(x[0] - ans[0]);
            for (int i = 0; i < (int)x.Count; i++)
            {
                var err = Math.Abs(x[i] - ans[i]);
                if (err > max_err)
                {
                    max_err = err;
                }
            }

            Console.WriteLine($"rr = {rr}");
            Console.WriteLine($"pp = {pp}");
            Console.WriteLine($"max err. = {max_err}");
            ulong overhead = (ulong)sw_gpu.ElapsedMilliseconds
                - (SparkCL.Core.IOTime + SparkCL.Core.KernTime) / 1_000_000 - (ulong)sw_total.ElapsedMilliseconds;
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"BiCGSTAB с OpenCL: {sw_gpu.ElapsedMilliseconds}мс");
            Console.WriteLine($"Время на операции:");
            Console.WriteLine($"IO: {SparkCL.Core.IOTime/1_000}мкс");
            Console.WriteLine($"Код OpenCL: {SparkCL.Core.KernTime/1_000_000}мс");
            Console.WriteLine($"Вычисления на хосте: {sw_total.ElapsedMilliseconds}мс");
            Console.WriteLine($"Накладные расходы: {overhead}мс");
        }

        static void BiCGSTAB_cpu()
        {
            // на Intel флаги не повлияли на производительность
            var mat =   LoadArray<Real>(File.OpenText("./mat"));
            var f =     LoadArray<Real>(File.OpenText("./f"));
            var aptr =  LoadArray<int> (File.OpenText("./aptr"));
            var jptr =  LoadArray<int> (File.OpenText("./jptr"));
            var x =     LoadArray<Real>(File.OpenText("./x"));
            var ans =   LoadArray<Real>(File.OpenText("./ans"));

            var r =     new List<Real>(new Real[x.Count]);
            var r_hat = new List<Real>(new Real[x.Count]);
            var p =     new List<Real>(new Real[x.Count]);
            var nu =    new List<Real>(new Real[x.Count]);
            var h =     new List<Real>(new Real[x.Count]);
            var s =     new List<Real>(new Real[x.Count]);
            var t =     new List<Real>(new Real[x.Count]);

            // var f32 = new SparkCL.Memory<Real>(1);

            var sw_host = new Stopwatch();
            sw_host.Start();

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
                if (Real.IsNaN(alpha))
                {
                    Console.WriteLine("альфа говно");
                    break;
                }

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
                    p[i] = r[i] + beta * (p[i] - w*nu[i]);

                });
                pp = pp1;
            }
            sw_host.Stop();

            Real max_err = Math.Abs(x[0] - ans[0]);
            for (int i = 0; i < (int)x.Count; i++)
            {
                var err = Math.Abs(x[i] - ans[i]);
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
        
        static void print(SparkCL.Memory<Real> mem, uint first)
        {
            for (uint i = 0; i < mem.Count && i < first; i++)
            {
                Console.WriteLine($"{mem[(int)i]}");
            }
            Console.WriteLine("...");
        }
        static void print(List<Real> mem, uint first)
        {
            for (uint i = 0; i < mem.Count && i < first; i++)
            {
                Console.WriteLine($"{mem[(int)i]}");
            }
            Console.WriteLine("...");
        }

        static void MyFor(int i0, int i1, Action<int> iteration)
        {
            if (HOST_PARALLEL)
            {
                Parallel.For(i0, i1, (i) =>
                {
                    iteration(i);
                });
            } else {
                for (int i = i0; i < i1; i++)
                {
                    iteration(i);
                }
            }
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

            var losPrepare1 = solvers.GetKernel(
                "LOS_prepare1",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                losPrepare1.PushArg(mat);
                losPrepare1.PushArg(aptr);
                losPrepare1.PushArg(jptr);
                losPrepare1.PushArg((uint)x.Count);
                losPrepare1.PushArg(f);
                losPrepare1.PushArg(x);
                losPrepare1.PushArg(r);

            var losPrepare2 = solvers.GetKernel(
                "LOS_prepare2",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                losPrepare2.PushArg(mat);
                losPrepare2.PushArg(aptr);
                losPrepare2.PushArg(jptr);
                losPrepare2.PushArg((uint)x.Count);
                losPrepare2.PushArg(r);
                losPrepare2.PushArg(p);

            var arMul = solvers.GetKernel(
                "MSRMul",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                arMul.PushArg(mat);
                arMul.PushArg(aptr);
                arMul.PushArg(jptr);
                arMul.PushArg((uint)x.Count);
                arMul.PushArg(r);
                arMul.PushArg(ar);

            var losXr = solvers.GetKernel(
                "LOS_xr",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                losXr.PushArg(f);
                losXr.PushArg(p);
                losXr.PushArg(0); // альфа
                losXr.PushArg(x);
                losXr.PushArg(r);

            var losFp = solvers.GetKernel(
                "LOS_fp",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                losFp.PushArg(r);
                losFp.PushArg(ar);
                losFp.PushArg(0); // бета
                losFp.PushArg(f);
                losFp.PushArg(p);

            var dot = solvers.GetKernel(
                "dot_kernel",
                globalWork: new(x.Count),
                localWork:  new(32)
            );
                dot.SetArg(2, (uint)x.Count);
                dot.SetArg(3, f32);
                dot.SetSize(4, (uint)32);

            ulong ioTime = 0;
            ulong kernelTime = 0;
            // LOS
            Real bb = f.Dot(f);
            
            var evKern = losPrepare1.Execute();
            var evIO = r.Read();
            ioTime += evIO.GetElapsed();
            kernelTime += evKern.GetElapsed();

            evKern = losPrepare2.Execute();
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
                evKern = losXr.Execute();
                evIO = r.Read();
                kernelTime += evKern.GetElapsed();
                ioTime += evIO.GetElapsed();

                evKern = arMul.Execute();
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
                evKern = losFp.Execute();
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
                if (Real.IsNaN(alpha))
                {
                    break;
                }
                
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
            Console.WriteLine($"Итераций: {iter}");
            Console.WriteLine($"max err: {max_err}");
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
