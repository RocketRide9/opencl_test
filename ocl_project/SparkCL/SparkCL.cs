using Silk.NET.OpenCL;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

// идея сократить область применения до вычисления на одном устройстве.
// это должно упростить использования OpenCL, абстрагируя понятия контекста,
// очереди команд и устройства.
namespace SparkCL
{
    using System.Globalization;
    using SparkOCL;
    static class StarterKit
    {
        // создать объекты на первом попавшемся GPU
        static public void GetStarterKit(
            out SparkOCL.Context context,
            out SparkOCL.Device device,
            out SparkOCL.CommandQueue commandQueue)
        {
            context = Context.FromType(DeviceType.Gpu);

            Platform.Get(out var platforms);
            var platform = platforms[0];

            platform.GetDevices(DeviceType.Gpu, out var devices);
            device = devices[0];

            commandQueue = new CommandQueue(context, device);
        }
    }

    static class Core
    {
        static internal Context? context;
        static internal CommandQueue? queue;
        static internal SparkOCL.Device? device;
        static public ulong IOTime { get; internal set; } = 0;
        static public ulong KernTime { get; internal set; } = 0;

        static public void Init()
        {
            context = Context.FromType(DeviceType.Gpu);

            Platform.Get(out var platforms);
            var platform = platforms[0];

            platform.GetDevices(DeviceType.Gpu, out var devices);
            device = devices[0];

            queue = new CommandQueue(context, device);
        }
    }

    /*
    class Device
    {
        static Context context;
        static CommandQueue queue;
        static SparkOCL.Device device;

        public Device()
        {
            context = Context.FromType(DeviceType.Gpu);

            Platform.Get(out var platforms);
            var platform = platforms[0];

            platform.GetDevices(DeviceType.Gpu, out var devices);
            device = devices[0];

            queue = new CommandQueue(context, device);
        }
    }
    */

    public static class EventExt
    {
        public static ulong GetElapsed(this SparkOCL.Event @event)
        {
            var s = @event.GetProfilingInfo(ProfilingInfo.Start);
            var c = @event.GetProfilingInfo(ProfilingInfo.End);

            return c - s;
        }
    }

    class Program
    {
        SparkOCL.Program program;

        public Program(string fileName)
        {
            program = SparkOCL.Program.FromFilename(Core.context!, Core.device!, fileName);
        }

        public SparkCL.Kernel GetKernel(string kernelName, NDRange globalWork, NDRange localWork)
        {
            var oclKernel = new SparkOCL.Kernel(program, kernelName);
            return new Kernel(oclKernel, globalWork, localWork);
        }
    }

    class Kernel
    {
        SparkOCL.Kernel kernel;
        public NDRange GlobalWork { get; set; }
        public NDRange LocalWork { get; set; }

        public Event Execute()
        {
            Core.queue!.EnqueueNDRangeKernel(kernel, new NDRange(), GlobalWork, LocalWork, out var ev);
            ev.Wait();
            Core.KernTime += ev.GetElapsed();
            return ev;
        }

        public uint PushArg<T>(
            SparkCL.Memory<T> buffer)
        where T: unmanaged, INumber<T>
        {
            return kernel.PushArg(buffer.buffer);
        }

        public uint PushArg<T>(
            T arg)
        where T: unmanaged
        {
            return kernel.PushArg(arg);
        }

        public void SetArg<T>(
            uint idx,
            T arg)
        where T: unmanaged
        {
            kernel.SetArg(idx, arg);
        }

        public void SetArg<T>(
            uint idx,
            SparkCL.Memory<T> mem)
        where T: unmanaged, INumber<T>
        {
            kernel.SetArg(idx, mem.buffer);
        }

        public void SetSize(
            uint idx,
            nuint sz)
        {
            kernel.SetSize(idx, sz);
        }

        internal Kernel(SparkOCL.Kernel kernel, NDRange globalWork, NDRange localWork)
        {
            this.kernel = kernel;
            GlobalWork = globalWork;
            LocalWork = localWork;
        }
    }

    public unsafe class Memory<T> : IDisposable
    where T: unmanaged, INumber<T> 
    {
        internal Buffer<T> buffer;
//        internal void* mappedPtr;
        internal Array<T> array;
        public nuint Count { get => array.Count; }

        public Memory(ReadOnlySpan<T> in_array, MemFlags flags = MemFlags.ReadWrite)
        {
            this.array = new(in_array);
            buffer = new(Core.context!, flags | MemFlags.UseHostPtr, this.array);
        }

        public Memory(StreamReader file, MemFlags flags = MemFlags.ReadWrite)
        {
            var sizeStr = file.ReadLine();
            var size = nuint.Parse(sizeStr!);
            this.array = new Array<T>(size);

            for (int i = 0; i < (int)size; i++)
            {
                var row = file.ReadLine();
                T elem;
                try {
                    elem = T.Parse(row!, CultureInfo.InvariantCulture);
                } catch (SystemException) {
                    throw new System.Exception($"i = {i}");
                }
                array[i] = elem;
            }
            buffer = new(Core.context!, flags | MemFlags.UseHostPtr, this.array);
        }

        public Memory (nuint size, MemFlags flags = MemFlags.ReadWrite)
        {
            array = new(size);
            buffer = new(Core.context!, flags | MemFlags.UseHostPtr, this.array);
        }

        public T this[int i]
        {
            get => array[i];
            set => array[i] = value;
        }

        //public Event Map(
        //    MapFlags flags,
        //    bool blocking = true
        //)
        //{
        //    mappedPtr = Core.queue!.EnqueueMapBuffer(buffer, blocking, flags, 0, array.Count, out var ev);
        //    return ev;
        //}

        //unsafe public Event Unmap()
        //{
        //    Core.queue!.EnqueueUnmapMemObject(buffer, mappedPtr, out var ev);
        //    return ev;
        //}

        public Event Read(
            bool blocking = true
        )
        {
            Core.queue!.EnqueueReadBuffer(buffer, blocking, 0, array, out var ev);
            Core.IOTime += ev.GetElapsed();
            return ev;
        }

        public Event Write(
            bool blocking = true
        )
        {
            Core.queue!.EnqueueWriteBuffer(buffer, blocking, 0, array, out var ev);
            Core.IOTime += ev.GetElapsed();
            return ev;
        }

        public T Dot(Memory<T> rhs)
        {
            T res = default;
            for (int i = 0; i < (int)Count; i++)
            {
                res += this[i] * rhs[i];
            }
            return res;
        }

        private bool disposedValue;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }
                array.Dispose();
                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~Memory()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
    }
}

// обёртка над Silk.NET.OpenCL для удобного использования в csharp
namespace SparkOCL
{
    internal static class CLHandle
    {
        static public CL Api = CL.GetApi();
    }

    public class Event
    {
        public nint Handle { get; }

        unsafe public ulong GetProfilingInfo(
            ProfilingInfo info
        )
        {
            var api = CLHandle.Api;

            ulong time;
            int err = api.GetEventProfilingInfo(Handle, info, 8, &time, null);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get profiling info, code: {err}");
            }

            return time;
        }
        
        unsafe public void Wait()
        {
            var api = CLHandle.Api;

            var handle = Handle;
            int err = api.WaitForEvents(1, &handle);
            
            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't wait for event, code: {err}");
            }
        }

        internal Event(nint h)
        {
            Handle = h;
        }

        ~Event()
        {
            var api = CLHandle.Api;
            api.ReleaseEvent(Handle);
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
            nint[] contextProperties =
            [
                (nint)ContextProperties.Platform,
                platforms[0].Handle,
                0
            ];

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

            platforms = new()
            {
                Capacity = (int)n
            };
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
            QueueProperties[] props = [
                (QueueProperties)CommandQueueInfo.Properties, (QueueProperties) CommandQueueProperties.ProfilingEnable,
                0
            ];
            fixed (QueueProperties *p = props)
            {
                Handle = api.CreateCommandQueueWithProperties(context.Handle, device.Handle, p, &err);
            }

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
            NDRange local,
            out Event @event)
        {
            var api = CLHandle.Api;

            int err;
            nint event_h;
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
                    &event_h);
            }
            @event = new Event(event_h);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue kernel, code: {err}");
            }
        }

        public unsafe void* EnqueueMapBuffer<T>(
            Buffer<T> buffer,
            bool blocking,
            MapFlags flags,
            nuint offset,
            nuint count,
            out Event @event)
        where T : unmanaged
        {
            var api = CLHandle.Api;
            nint event_h;
            var ptr = api.EnqueueMapBuffer(
                Handle,
                buffer.Handle,
                blocking,
                flags,
                offset,
                count * (nuint) sizeof(T),
                0,
                null,
                out event_h,
                out int err);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue buffer map, code: {err}");
            }
            @event = new Event(event_h);

            return ptr;
        }

        public unsafe void EnqueueReadBuffer<T>(
            Buffer<T> buffer,
            bool blocking,
            nuint offset,
            Array<T> array,
            out Event @event)
        where T : unmanaged
        {
            var api = CLHandle.Api;

            nint event_h;
            int err = api.EnqueueReadBuffer(
                Handle,
                buffer.Handle,
                blocking,
                offset,
                array.Count * (nuint) sizeof(T),
                array.Buf,
                0,
                null,
                out event_h);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue buffer read, code: {err}");
            }
            @event = new Event(event_h);
        }

        public unsafe void EnqueueWriteBuffer<T>(
            Buffer<T> buffer,
            bool blocking,
            nuint offset,
            Array<T> array,
            out Event @event)
        where T : unmanaged
        {
            var api = CLHandle.Api;

            nint event_h;
            int err = api.EnqueueWriteBuffer(
                Handle,
                buffer.Handle,
                blocking,
                offset,
                array.Count * (nuint) sizeof(T),
                array.Buf,
                0,
                null,
                out event_h);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue buffer read, code: {err}");
            }
            @event = new Event(event_h);
        }

        public unsafe void EnqueueUnmapMemObject<T>(
            Buffer<T> buffer,
            void *ptr,
            out Event @event)
        where T : unmanaged
        {
            var api = CLHandle.Api;

            nint event_h;
            int err = api.EnqueueUnmapMemObject(
                Handle,
                buffer.Handle,
                ptr,
                0,
                null,
                out event_h);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't enqueue memory object unmap, code: {err}");
            }
            @event = new Event(event_h);
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
        uint argCount = 0;

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

        public uint PushArg<T>(
            SparkOCL.Buffer<T> buffer)
        where T: unmanaged
        {
            SetArg(argCount, buffer);
            argCount++;
            return argCount;
        }

        public uint PushArg<T>(
            T arg)
        where T: unmanaged
        {
            SetArg(argCount, arg);
            argCount++;
            return argCount;
        }

        unsafe public void SetArg<T>(
            uint arg_index,
            SparkOCL.Buffer<T> buffer)
        where T : unmanaged
        {
            var api = CLHandle.Api;
            var binding = buffer.Handle;

            int err = api.SetKernelArg(Handle, arg_index, (nuint)sizeof(nint), ref binding);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to set kernel argument (index = {arg_index}), code: {err}");
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
                throw new System.Exception($"Failed to set kernel argument, code: {err}, arg size: {(nuint)sizeof(T)}");
            }
        }

        unsafe public void SetSize(
            uint arg_index,
            nuint sz)
        {
            var api = CLHandle.Api;

            int err = api.SetKernelArg(Handle, arg_index, (nuint)sizeof(float) * sz, null);
            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to set kernel argument, code: {err}");
            }
        }
    }

    class Buffer<T>
    where T : unmanaged
    {
        public nint Handle { get; }

        unsafe public Buffer(Context context, MemFlags flags, SparkOCL.Array<T> array)
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

    unsafe class Array<T> : IDisposable
    where T: unmanaged
    {
        public T* Buf { get; internal set; }
        public nuint Count { get; }
        public nuint ElementSize { get; }

        public Array (ReadOnlySpan<T> array)
        {
            ElementSize = (nuint)sizeof(T);
            Buf = (T*)NativeMemory.AlignedAlloc((nuint)array.Length * ElementSize, 4096);
            this.Count = (nuint)array.Length;
            array.CopyTo(new Span<T>(Buf, array.Length));
        }

        public Array (nuint size)
        {
            ElementSize = (nuint)sizeof(T);
            Buf = (T*)NativeMemory.AlignedAlloc(size * ElementSize, 4096);
            this.Count = size;
        }

        public T this[int i]
        {
            get
            {
                return Buf[i];
            }
            set
            {
                Buf[i] = value;
            }
        }


        private bool disposedValue;

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects)
                }

                NativeMemory.AlignedFree(Buf);
                // TODO: free unmanaged resources (unmanaged objects) and override finalizer
                // TODO: set large fields to null
                disposedValue = true;
            }
        }

        // TODO: override finalizer only if 'Dispose(bool disposing)' has code to free unmanaged resources
        ~Array()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
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
