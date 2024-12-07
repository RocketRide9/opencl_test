using Silk.NET.OpenCL;
using Quasar.Native;

using System;
using System.Numerics;
using System.IO;
using System.Globalization;

// идея сократить область применения до вычисления на одном устройстве.
// это должно упростить использования OpenCL, абстрагируя понятия контекста,
// очереди команд и устройства.
namespace SparkCL
{
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

    static public class Core
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
        uint lastPushed = 0;

        public Event Execute()
        {
            Core.queue!.EnqueueNDRangeKernel(kernel, new NDRange(), GlobalWork, LocalWork, out var ev);
            // HACK: ожидание завершения выполнения ядра для замера времени.
            // Замедление работы не было обнаружено, но оно может возникнуть
            // при постановке на очередь нескольких ядер
            ev.Wait();
            Core.KernTime += ev.GetElapsed();
            return ev;
        }

        public uint PushArg<T>(
            SparkCL.Memory<T> mem)
        where T: unmanaged, INumber<T>
        {
            kernel.SetArg(lastPushed, mem.buffer);
            lastPushed++;
            return lastPushed;
        }

        public uint PushArg<T>(
            T arg)
        where T: unmanaged
        {
            kernel.SetArg(lastPushed, arg);
            lastPushed++;
            return lastPushed;
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

        public Event CopyTo(
            Memory<T> destination
        )
        {
            Core.queue!.EnqueueCopyBuffer(buffer, destination.buffer, 0, 0, Count, out var ev);
            ev.Wait();
            Core.IOTime += ev.GetElapsed();
            return ev;
        }

        public T Dot(Memory<T> rhs)
        {
            // float res = (float)BLAS.dot(
            //     (int) this.Count,
            //     new ReadOnlySpan<float>(    array.Buf, (int)Count),
            //     new ReadOnlySpan<float>(rhs.array.Buf, (int)Count)
            // );
            // return res;

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
