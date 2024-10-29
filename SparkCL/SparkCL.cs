using Silk.NET.OpenCL;
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;


namespace SparkCL
{
    internal static class CLHandle
    {
        static public CL Api = CL.GetApi();
    }

    static class StarterKit
    {
        // создать объекты на первом попавшемся GPU
        static public void GetStarterKit (out Context context, out Device device, out CommandQueue commandQueue)
        {
            context = Context.FromType(DeviceType.Gpu);

            Platform.Get(out var platforms);
            var platform = platforms[0];

            platform.GetDevices(DeviceType.Gpu, out var devices);
            device = devices[0];

            commandQueue = new CommandQueue(context, device);
        }
    }

    class Context
    {
        public nint Handle { get; }

        unsafe static public Context FromDevice(
            Device device)
        {
            var api = CLHandle.Api;
            var device_h = device.Handle;
            int err_p;
            var h = api.CreateContext(null, 1, &device_h, null, null, &err_p);

            if (err_p != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't create context on requested device, code: {err_p}");
            }

            var res = new Context(h);
            return res;
        }

        unsafe static public Context FromType(
            DeviceType type)
        {
            var api = CLHandle.Api;

            var platforms = new List<Platform>();
            Platform.Get(out platforms);

            // Next, create an OpenCL context on the platform.  Attempt to
            // create a GPU-based context, and if that fails, try to create
            // a CPU-based context.
            nint[] contextProperties = new nint[]
            {
                (nint)ContextProperties.Platform,
                platforms[0].Handle,
                0
            };

            fixed (nint* p = contextProperties)
            {
                int errNum;
                var context_handle = api.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
                if (errNum != (int) ErrorCodes.Success)
                {
                    throw new System.Exception($"Couldn't create context on requested device type, code: {errNum}");
                }

                return new Context(context_handle);
            }
        }

        private Context(nint h)
        {
            Handle = h;
        }

        ~Context()
        {
            CLHandle.Api.ReleaseContext(Handle);
        }
    }

    class Platform
    {
        public nint Handle { get; }
        unsafe public static void Get(out List<Platform> platforms)
        {
            uint n = 0;
            var api = CLHandle.Api;
            var err = api.GetPlatformIDs(0, null, &n);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get platform ids, code: {err}");
            }

            var ids = new nint[n];
            err = api.GetPlatformIDs(n, ids, (uint *)null);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get platform ids, code: {err}");
            }

            platforms = new();
            platforms.Capacity = (int) n;
            for (int i = 0; i < n; i++)
            {
                var p = new Platform(ids[i]);
                platforms.Add(p);
            }
        }

        unsafe public void GetDevices(DeviceType type, out List<Device> devices)
        {
            uint n = 0;
            var api = CLHandle.Api;
            var err = api.GetDeviceIDs(Handle, type, 0, null, &n);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get devices ID, code: {err}");
            }

            var ids = new nint[n];
            err = api.GetDeviceIDs(Handle, type, n, ids, (uint *)null);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get devices ID, code: {err}");
            }

            devices = new((int) n);
            for (int i = 0; i < n; i++)
            {
                var p = new Device(ids[i]);
                devices.Add(p);
            }
        }

        private Platform(nint h)
        {
            Handle = h;
        }
    }

    class Device
    {
        DeviceType Type { get; }
        public nint Handle { get; }

        internal Device(nint h)
        {
            Handle = h;
        }

        ~Device()
        {
            CLHandle.Api.ReleaseDevice(Handle);
        }
    }


    class NDRange
    {
        public uint Dimensions { get; }
        public nuint[] Sizes { get; } = [1, 1, 1];

        public NDRange()
        {
            Dimensions = 0;
            Sizes[0] = 0;
            Sizes[1] = 0;
            Sizes[2] = 0;
        }
        public NDRange(nuint size0)
        {
            Dimensions = 1;
            Sizes[0] = size0;
            Sizes[1] = 1;
            Sizes[2] = 1;
        }
        public NDRange(
            nuint size0,
            nuint size1)
        {
            Dimensions = 2;
            Sizes[0] = size0;
            Sizes[1] = size1;
            Sizes[2] = 1;
        }
        public NDRange(
            nuint size0,
            nuint size1,
            nuint size2)
        {
            Dimensions = 1;
            Sizes[0] = size0;
            Sizes[1] = size1;
            Sizes[2] = size2;
        }

        nuint this[int i]
        {
            get => Sizes[i];
        }
    }

    class CommandQueue
    {
        public nint Handle { get; }

        public unsafe CommandQueue(Context context, Device device)
        {
            var api = CLHandle.Api;

            int err;
            Handle = api.CreateCommandQueue(context.Handle, device.Handle, CommandQueueProperties.None, &err);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't create command queue, code: {err}");
            }
        }

        public void Finish()
        {
            int err = CLHandle.Api.Finish(Handle);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't finish command queue, code: {err}");
            }
        }

        public unsafe void EnqueueNDRangeKernel(
            Kernel kernel,
            NDRange offset,
            NDRange global,
            NDRange local)
        {
            var api = CLHandle.Api;

            int err;
            fixed (nuint *g = global.Sizes)
            fixed (nuint *o = offset.Sizes)
            {
                err = api.EnqueueNdrangeKernel(
                    Handle,
                    kernel.Handle,
                    global.Dimensions,
                    offset.Dimensions != 0 ? o : null,
                    g,
                    null,
                    0,
                    null,
                    null);
            }
            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue kernel, code: {err}");
            }
        }

        public unsafe Span<T> EnqueueMapBuffer<T>(
            Buffer<T> buffer,
            bool blocking,
            MapFlags flags,
            nuint offset,
            nuint count)
        where T : unmanaged
        {
            var api = CLHandle.Api;

            var ptr = api.EnqueueMapBuffer(
                Handle,
                buffer.Handle,
                blocking,
                flags,
                offset,
                count * (nuint) sizeof(T),
                0,
                null,
                null,
                out int err);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue buffer map, code: {err}");
            }

            return new Span<T>(ptr, (int) count);
        }

        private CommandQueue(nint h)
        {
            Handle = h;
        }

        ~CommandQueue()
        {
            CLHandle.Api.ReleaseCommandQueue(Handle);
        }
    }

    class Kernel
    {
        public nint Handle { get; }

        unsafe public Kernel(
            Program program,
            string name)
        {
            var api = CLHandle.Api;
            int err;
            Handle = api.CreateKernel(program.Handle, Encoding.ASCII.GetBytes(name), &err);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create kernel, code: {err}");
            }
        }

        ~Kernel()
        {
            CLHandle.Api.ReleaseKernel(Handle);
        }

        unsafe public void SetArg<T>(
            uint arg_index,
            SparkCL.Buffer<T> buffer)
        where T : unmanaged
        {
            var api = CLHandle.Api;
            var binding = buffer.Handle;

            int err = api.SetKernelArg(Handle, arg_index, (nuint)sizeof(nint), ref binding);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to set kernel argument, code: {err}");
            }
        }

        unsafe public void SetArg<T>(
            uint arg_index,
            T arg)
        where T: unmanaged
        {
            var api = CLHandle.Api;

            int err = api.SetKernelArg(Handle, arg_index, (nuint)sizeof(T), ref arg);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to set kernel argument, code: {err}");
            }
        }
    }

    class Buffer<T>
    where T : unmanaged
    {
        public nint Handle { get; }

        unsafe public Buffer(Context context, MemFlags flags, SparkCL.Array<T> array)
        {
            var api = CLHandle.Api;
            int err;
            Handle = api.CreateBuffer(context.Handle, flags, (nuint) sizeof(T) * array.Count, array.Buf, &err);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create buffer, code: {err}");
            }
        }

        ~Buffer()
        {
            CLHandle.Api.ReleaseMemObject(Handle);
        }
    }

    unsafe class Array<T>
    where T: unmanaged
    {
        public void* Buf { get; }
        public nuint Count { get; }
        public nuint ElementSize { get; }

        public Array (nuint size)
        {
            ElementSize = (nuint)sizeof(T);
            Buf = NativeMemory.AlignedAlloc(size * ElementSize, 4096);
            this.Count = size;
        }

        // public Span<T> GetSpan ()
        // {
        //     return new Span<T>(buf, (int)size);
        // }

        public T this[int i]
        {
            get
            {
                var sp = new Span<T>(Buf, (int)Count);
                return sp[i];
            }
            set
            {
                var sp = new Span<T>(Buf, (int)Count);
                sp[i] = value;
            }
        }

        ~Array()
        {
            NativeMemory.AlignedFree(Buf);
        }
    }

    class Program
    {
        internal nint Handle { get; }

        unsafe static public Program FromFilename(
            Context context,
            Device device,
            string fileName)
        {
            var api = CLHandle.Api;
            using StreamReader sr = new StreamReader(fileName);
            string clStr = sr.ReadToEnd();

            int err;
            var program = api.CreateProgramWithSource(context.Handle, 1, new string[] { clStr }, null, &err);
            if (program == IntPtr.Zero || err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create CL program from source, code: {err}");
            }

            var errNum = api.BuildProgram(program, 0, null, (byte*)null, null, null);

            if (errNum != (int)ErrorCodes.Success)
            {
                _ = api.GetProgramBuildInfo(program, device.Handle, ProgramBuildInfo.BuildLog, 0, null, out nuint buildLogSize);
                byte[] log = new byte[buildLogSize / (nuint)sizeof(byte)];
                fixed (void* pValue = log)
                {
                    api.GetProgramBuildInfo(program, device.Handle, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
                }
                string? build_log = System.Text.Encoding.UTF8.GetString(log);

                //Console.WriteLine("Error in kernel: ");
                Console.WriteLine("=============== OpenCL Program Build Info ================");
                Console.WriteLine(build_log);
                Console.WriteLine("==========================================================");

                api.ReleaseProgram(program);
                throw new Exception($"OpenCL build failed.");
            }

            return new Program(program);
        }

        Program(nint h)
        {
            Handle = h;
        }

        ~Program()
        {
            CLHandle.Api.ReleaseProgram(Handle);
        }
    }
}
