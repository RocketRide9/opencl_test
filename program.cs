// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Silk.NET.OpenCL;
using System;
using System.IO;
using System.Diagnostics;
using System.Threading.Tasks;
using real = System.Single;
using SparkCL;

namespace HelloWorld
{
    internal class Program
    {
        // наибольшая матрица
        // для 4ГиБ - 32320 x 32320
        // для 1ГиБ - 16384 x 16384
        const int M_ROWS = 32320;
        const int M_COLS = 32320;

        // P - кол-во потоков для счёта скалярного произведения
        // MT - кол-во скалярных произведений, считающихся параллельно
        // P * MT - размер локальной рабочей группы
        const nuint P = 1;
        const nuint MT = 16;
        static void Main(string[] args)
        {
            Random rand = new Random();

            SparkCL.StarterKit.GetStarterKit(out var context, out var device, out var commandQueue);

            // Create OpenCL program from HelloWorld.cl kernel source
            var program = SparkCL.Program.FromFilename(context, device, "HelloWorld.cl");

            // Create memory objects that will be used as arguments to
            // kernel.  First create host memory arrays that will be
            // used to store the arguments to the kernel
            var mat = new SparkCL.Array<real>(M_COLS * M_ROWS);
            var vec = new SparkCL.Array<real>(M_COLS);
            var res = new SparkCL.Array<real>(M_ROWS);

            for (int j = 0; j < M_COLS; j++)
            {
                for (int i = 0; i < M_ROWS; i++)
                {
                    mat[i + j * M_ROWS] = 1;
                }
            }
            for (int j = 0; j < M_COLS; j++)
            {
                vec[j] = 1;
            }
            Console.WriteLine($"Матрица {M_ROWS} x {M_COLS}");
            Console.WriteLine($"MT = {MT}");
            var sw_gpu = new Stopwatch();
            sw_gpu.Start();

            var matBuffer = new SparkCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, mat);
            var vecBuffer = new SparkCL.Buffer<real>(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, vec);
            var resBuffer = new SparkCL.Buffer<real>(context, MemFlags.WriteOnly | MemFlags.UseHostPtr, res);

            var globalWork = new NDRange(M_ROWS);
            var localWork = new NDRange(MT);

            // Create OpenCL kernel
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

            commandQueue.Finish();

            sw_gpu.Stop();

            for (int j = 0; j < M_COLS; j++)
            {
                for (int i = 0; i < M_ROWS; i++)
                {
                    res[i] -= mat[i + j * M_ROWS] * vec[j];
                }
            }
            var err = (real)0;
            for (var j = 0; j < M_ROWS; j++)
            {
                var res2 = res_mapped[j];
                if (Math.Abs(res2) > err)
                {
                    err = res2;
                }
            }

            var sw_cpu = new Stopwatch();

            var mat_cpu = new real[M_ROWS, M_ROWS];
            var right_cpu = new real[M_COLS];
            var ans_cpu = new real[M_COLS];

            for (int j = 0; j < M_COLS; j++)
            {
                {
                for (int i = 0; i < M_ROWS; i++)
                    mat_cpu[j, i] = 1;
                }
            }
            for (int j = 0; j < M_COLS; j++)
            {
                right_cpu[j] = 1;
            }


            sw_cpu.Start();
            Parallel.For(0, M_ROWS, (Action<int, ParallelLoopState>)((i, state) =>
            {
                var acc = (real)0;
                for (int j = 0; j < M_COLS; j++)
                {
                    acc += mat_cpu[i, j] * right_cpu[j];
                }
                ans_cpu[i] = acc;
            }));
            sw_cpu.Stop();

            Console.WriteLine($"время CPU: {sw_cpu.ElapsedMilliseconds}");
            Console.WriteLine($"Устройство OCL: {sw_gpu.ElapsedMilliseconds}");
            Console.WriteLine($"Макс. погрешность OCL: {err}");
        }

        /// <summary>
        /// Cleanup any created OpenCL resources
        /// </summary>
        /// <param name="cl"></param>
        /// <param name="context"></param>
        /// <param name="commandQueue"></param>
        /// <param name="program"></param>
        /// <param name="kernel"></param>
        /// <param name="memObjects"></param>
        static void Cleanup(CL cl, nint context, nint commandQueue,
             nint program, nint kernel, nint[] memObjects)
        {
            for (int i = 0; i < memObjects.Length; i++)
            {
                if (memObjects[i] != 0)
                    cl.ReleaseMemObject(memObjects[i]);
            }
            if (commandQueue != 0)
                cl.ReleaseCommandQueue(commandQueue);

            if (kernel != 0)
                cl.ReleaseKernel(kernel);

            if (program != 0)
                cl.ReleaseProgram(program);

            if (context != 0)
                cl.ReleaseContext(context);
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
