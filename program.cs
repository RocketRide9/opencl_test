// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Silk.NET.OpenCL;
using System;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;
using real = System.Single;
using System.Threading.Tasks;

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
        const nuint MT = 64;
        static unsafe void Main(string[] args)
        {
            var cl = CL.GetApi();
            nint context = 0;
            nint commandQueue = 0;
            nint program = 0;
            nint device = 0;
            Random rand = new Random();

            nint[] memObjects = new nint[3];

            // Create an OpenCL context on first available platform
            context = CreateContext(cl);
            if (context == IntPtr.Zero)
            {
                Console.WriteLine("Failed to create OpenCL context.");
                return;
            }
            // Create a command-queue on the first device available
            // on the created context
            commandQueue = CreateCommandQueue(cl, context, ref device);
            if (commandQueue == IntPtr.Zero)
            {
                Cleanup(cl, context, commandQueue, program, 0, memObjects);
                return;
            }

            // Create OpenCL program from HelloWorld.cl kernel source
            program = CreateProgram(cl, context, device, "HelloWorld.cl");
            if (program == IntPtr.Zero)
            {
                Cleanup(cl, context, commandQueue, program, 0, memObjects);
                return;
            }

            // Create OpenCL kernel
            int err_code;
            nint kernel = cl.CreateKernel(program, "gemv3", out err_code);
            if (kernel == IntPtr.Zero)
            {
                Console.WriteLine($"Failed to create kernel: {err_code}");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            nint kernel_reduce = cl.CreateKernel(program, "reduce_rows", out err_code);
            if (kernel_reduce == IntPtr.Zero)
            {
                Console.WriteLine($"Failed to create kernel: {err_code}");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            // Create memory objects that will be used as arguments to
            // kernel.  First create host memory arrays that will be
            // used to store the arguments to the kernel
            void* result_p = NativeMemory.AlignedAlloc(M_ROWS * sizeof(real), 4096);

            void* m_p = NativeMemory.AlignedAlloc((uint)M_COLS * M_ROWS * sizeof(real), 4096);
            var m = new Span<real>(m_p, (int)M_COLS * M_ROWS);

            void* b_p = NativeMemory.AlignedAlloc((uint)M_COLS * sizeof(real), 4096);
            var b = new Span<real>(b_p, (int)M_COLS);

            // промежуточная матрица для хранения частичных скалярных произведений
            void* partial_dots_p = NativeMemory.AlignedAlloc((uint)M_ROWS * P * sizeof(real), 4096);
            var partial_dots = new Span<real>(partial_dots_p, (int)(M_ROWS * P));

            for (int j = 0; j < M_COLS; j++)
            {
                {
                    for (int i = 0; i < M_ROWS; i++)
                        m[i + j * M_ROWS] = 1;
                }
            }
            for (int j = 0; j < M_COLS; j++)
            {
                b[j] = 1;
            }
            Console.WriteLine($"Матрица {M_ROWS} x {M_COLS}");
            Console.WriteLine($"MT = {MT}, P = {P}");
            var sw_gpu = new Stopwatch();
            sw_gpu.Start();

            memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, (nuint)sizeof(real) * M_COLS * M_ROWS, m_p, null);
            memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.UseHostPtr, sizeof(real) * M_COLS, b_p, null);
            memObjects[2] = cl.CreateBuffer(context, MemFlags.WriteOnly | MemFlags.UseHostPtr, sizeof(real) * M_ROWS, result_p, null);
            nint partial_dots_obj = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.UseHostPtr, sizeof(real) * M_ROWS * P, partial_dots_p, null);

            if (memObjects[0] == IntPtr.Zero || memObjects[1] == IntPtr.Zero || memObjects[2] == IntPtr.Zero)
            {
                Console.WriteLine("Error creating memory objects.");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            nuint[] globalWorkSize = new nuint[2] { M_ROWS, P };
            nuint[] localWorkSize = new nuint[2] { MT, P };

            int errNum = 0;
            {
                var cols = M_COLS;
                var rows = M_ROWS;
                var p = P;
                errNum |= cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), ref memObjects[0]);
                errNum |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint), ref memObjects[1]);
                errNum |= cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint), ref partial_dots_obj);
                errNum |= cl.SetKernelArg(kernel, 3, (nuint)sizeof(int), ref cols);
                errNum |= cl.SetKernelArg(kernel, 4, (nuint)sizeof(int), ref rows);
                errNum |= cl.SetKernelArg(kernel, 5, (nuint)sizeof(real) * localWorkSize[0] * localWorkSize[1], null);

                errNum |= cl.SetKernelArg(kernel_reduce, 0, (nuint)sizeof(nint), ref partial_dots_obj);
                errNum |= cl.SetKernelArg(kernel_reduce, 1, (nuint)sizeof(int), ref rows);
                errNum |= cl.SetKernelArg(kernel_reduce, 2, (nuint)sizeof(int), ref p);
            }

            if (errNum != (int)ErrorCodes.Success)
            {
                Console.WriteLine($"Error setting kernel arguments: {errNum}");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            // Queue the kernel up for execution across the array
            errNum = cl.EnqueueNdrangeKernel(
                    commandQueue,
                    kernel,
                    2,
                    (nuint*)null,
                    globalWorkSize,
                    localWorkSize,
                    0,
                    (nint*)null,
                    (nint*)null
                );
            if (errNum != (int)ErrorCodes.Success)
            {
                Console.WriteLine($"Error queuing kernel for execution: {errNum}");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            {
                nuint one = 1;
                nuint rows = M_ROWS;
                errNum = cl.EnqueueNdrangeKernel(
                        commandQueue,
                        kernel_reduce,
                        1,
                        (nuint*)null,
                        &rows,
                        &one,
                        0,
                        (nint*) null,
                        (nint*) null
                    );
            }

            if (errNum != (int) ErrorCodes.Success)
            {
                Console.WriteLine($"Error queuing kernel for execution: {errNum}");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }

            void* buf = cl.EnqueueMapBuffer(
                    commandQueue,
                    partial_dots_obj,
                    true,
                    MapFlags.Read,
                    0,
                    M_COLS * sizeof(real),
                    0,
                    null,
                    null,
                    out errNum
                );

            if (errNum != (int) ErrorCodes.Success)
            {
                Console.WriteLine("Error reading result buffer.");
                Cleanup(cl, context, commandQueue, program, kernel, memObjects);
                return;
            }
            cl.Finish(commandQueue);
            sw_gpu.Stop();

            var result = new Span<real>(buf, (int)(M_ROWS * P));
            for (int j = 0; j < M_COLS; j++)
            {
                for (int i = 0; i < M_ROWS; i++)
                {
                    result[i] -= m[i + j * M_ROWS] * b[j];
                }
            }
            var err = (real)0;
            for (var j = 0; j < M_ROWS; j++)
            {
                var res = result[j];
                if (Math.Abs(res) > err)
                {
                    err = res;
                }
            }
            Cleanup(cl, context, commandQueue, program, kernel, memObjects);
            NativeMemory.AlignedFree(b_p);
            NativeMemory.AlignedFree(result_p);
            NativeMemory.AlignedFree(m_p);

            var sw_cpu = new Stopwatch();

            var mat = new real[M_ROWS, M_ROWS];
            var right = new real[M_COLS];
            var ans = new real[M_COLS];

            for (int j = 0; j < M_COLS; j++)
            {
                {
                    for (int i = 0; i < M_ROWS; i++)
                        mat[j, i] = 1;
                }
            }
            for (int j = 0; j < M_COLS; j++)
            {
                right[j] = 1;
            }


            sw_cpu.Start();
            Parallel.For(0, M_ROWS, (i, state) =>
            {
                var acc = (real)0;
                for (int j = 0; j < M_COLS; j++)
                {
                    acc += mat[i, j] * right[j];
                }
                ans[i] = acc;
            });
            sw_cpu.Stop();

            Console.WriteLine($"время CPU: {sw_cpu.ElapsedMilliseconds}");
            Console.WriteLine($"Устройство OCL: {sw_gpu.ElapsedMilliseconds}");
            Console.WriteLine($"Макс. погрешность OCL: {err}");
        }

        /// <summary>
        /// Create an OpenCL program from the kernel source file
        /// </summary>
        /// <param name="cl"></param>
        /// <param name="context"></param>
        /// <param name="device"></param>
        /// <param name="fileName"></param>
        /// <returns></returns>
        static unsafe nint CreateProgram(CL cl, nint context, nint device, string fileName)
        {
            if (!File.Exists(fileName))
            {
                Console.WriteLine($"File does not exist: {fileName}");
                return IntPtr.Zero;
            }
            using StreamReader sr = new StreamReader(fileName);
            string clStr = sr.ReadToEnd();

            var program = cl.CreateProgramWithSource(context, 1, new string[] { clStr }, null, null);
            if (program == IntPtr.Zero)
            {
                Console.WriteLine("Failed to create CL program from source.");
                return IntPtr.Zero;
            }

            var errNum = cl.BuildProgram(program, 0, null, (byte*) null, null, null);

            if (errNum != (int) ErrorCodes.Success)
            {
                _ = cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0, null, out nuint buildLogSize);
                byte[] log = new byte[buildLogSize / (nuint) sizeof(byte)];
                fixed (void* pValue = log)
                {
                    cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
                }
                string? build_log = System.Text.Encoding.UTF8.GetString(log);

                //Console.WriteLine("Error in kernel: ");
                Console.WriteLine("=============== OpenCL Program Build Info ================");
                Console.WriteLine(build_log);
                Console.WriteLine("==========================================================");

                cl.ReleaseProgram(program);
                return IntPtr.Zero;
            }

            return program;
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

        /// <summary>
        /// Create a command queue on the first device available on the
        /// context
        /// </summary>
        /// <param name="cL"></param>
        /// <param name="context"></param>
        /// <param name="device"></param>
        /// <returns></returns>
        static unsafe nint CreateCommandQueue(CL cL, nint context, ref nint device)
        {
            int errNum = cL.GetContextInfo(context, ContextInfo.Devices, 0, null, out nuint deviceBufferSize);
            if (errNum != (int) ErrorCodes.Success)
            {
                Console.WriteLine("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
                return IntPtr.Zero;
            }

            if (deviceBufferSize <= 0)
            {
                Console.WriteLine("No devices available.");
                return IntPtr.Zero;
            }

            nint[]? devices = new nint[deviceBufferSize / (nuint) sizeof(nuint)];
            fixed (void* pValue = devices)
            {
                int er = cL.GetContextInfo(context, ContextInfo.Devices, deviceBufferSize, pValue, null);

            }
            if (errNum != (int) ErrorCodes.Success)
            {
                devices = null;
                Console.WriteLine("Failed to get device IDs");
                return IntPtr.Zero;
            }

            // In this example, we just choose the first available device.  In a
            // real program, you would likely use all available devices or choose
            // the highest performance device based on OpenCL device queries
            var commandQueue = cL.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, null);
            if (commandQueue == IntPtr.Zero)
            {
                Console.WriteLine("Failed to create commandQueue for device 0");
                return IntPtr.Zero;
            }

            device = devices[0];
            var name = GetDeviceInfo(cL, device, DeviceInfo.Name);
            Console.WriteLine($"Device name: {name}");
            var workGroup = GetDeviceInfo(cL, device, DeviceInfo.MaxWorkGroupSize);
            Console.WriteLine($"Max work group size: {workGroup}");
            var maxAllocSize = GetDeviceInfo(cL, device, DeviceInfo.MaxMemAllocSize);
            Console.WriteLine($"Max memory allocation size: {maxAllocSize}");
            // var exts = GetDeviceInfo(cL, device, DeviceInfo.Extensions);
            // Console.WriteLine($"Available extensions: {exts}");
            return commandQueue;
        }

        static unsafe string? GetDeviceInfo(CL cl, nint device, DeviceInfo infoEnum)
        {
            cl.GetDeviceInfo(device, infoEnum, 0, null, out nuint nameSize);
            byte[] infoBytes = new byte[nameSize / (nuint) sizeof(byte)];
            fixed (void *p = infoBytes)
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

        /// <summary>
        /// Create an OpenCL context on the first available platform using
        /// either a GPU or CPU depending on what is available.
        /// </summary>
        /// <param name="cL"></param>
        /// <returns></returns>
        static unsafe nint CreateContext(CL cL)
        {
            var errNum = cL.GetPlatformIDs(1, out nint firstPlatformId, out uint numPlatforms);
            if (errNum != (int) ErrorCodes.Success || numPlatforms <= 0)
            {
                Console.WriteLine("Failed to find any OpenCL platforms.");
                return IntPtr.Zero;
            }

            // Next, create an OpenCL context on the platform.  Attempt to
            // create a GPU-based context, and if that fails, try to create
            // a CPU-based context.
            nint[] contextProperties = new nint[]
            {
                (nint)ContextProperties.Platform,
                firstPlatformId,
                0
            };

            fixed (nint* p = contextProperties)
            {
                var context = cL.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
                if (errNum != (int) ErrorCodes.Success)
                {
                    Console.WriteLine("Could not create GPU context, trying CPU...");

                    context = cL.CreateContextFromType(p, DeviceType.Cpu, null, null, out errNum);

                    if (errNum != (int) ErrorCodes.Success)
                    {
                        Console.WriteLine("Failed to create an OpenCL GPU or CPU context.");
                        return IntPtr.Zero;
                    }

                    return context;
                }

                return context;
            }
        }
    }
}
