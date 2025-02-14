using Silk.NET.OpenCL;
using System;
using System.Numerics;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using static SparkOCL.CLHandle;
using System.Linq;

// обёртка над Silk.NET.OpenCL для удобного использования в csharp
namespace SparkOCL
{
    internal static class CLHandle
    {
        static public CL OCL = CL.GetApi();
    }

    public class Event: IDisposable
    {
        private bool disposedValue;

        public nint Handle { get; }

        unsafe public ulong GetProfilingInfo(
            ProfilingInfo info
        )
        {
            ulong time;
            int err = OCL.GetEventProfilingInfo(Handle, info, 8, &time, null);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get profiling info, code: {err}");
            }

            return time;
        }
        
        unsafe public void Wait()
        {

            var handle = Handle;
            int err = OCL.WaitForEvents(1, &handle);
            
            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't wait for event, code: {err}");
            }
        }

        internal Event(nint h)
        {
            Handle = h;
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
                OCL.ReleaseEvent(Handle);
                disposedValue = true;
            }
        }

        // TODO: переопределить метод завершения, только если "Dispose(bool disposing)" содержит код для освобождения неуправляемых ресурсов
        ~Event()
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

    class Context
    {
        public nint Handle { get; }

        unsafe static public Context FromDevice(
            Device device)
        {
            var device_h = device.Handle;
            int err_p;
            var h = OCL.CreateContext(null, 1, &device_h, null, null, &err_p);

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

            var platforms = new List<Platform>();
            Platform.Get(out platforms);
            
            nint[] contextProperties =
            [
                (nint)ContextProperties.Platform,
                platforms[0].Handle,
                0
            ];

            fixed (nint* p = contextProperties)
            {
                int errNum;
                var context_handle = OCL.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
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
            CLHandle.OCL.ReleaseContext(Handle);
        }
    }

    class Platform
    {
        public nint Handle { get; }
        unsafe public static void Get(out List<Platform> platforms)
        {
            uint n = 0;
            var err = OCL.GetPlatformIDs(0, null, &n);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get platform ids, code: {err}");
            }

            var ids = new nint[n];
            err = OCL.GetPlatformIDs(n, ids, (uint *)null);
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
            var err = OCL.GetDeviceIDs(Handle, type, 0, null, &n);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't get devices ID, code: {err}");
            }

            var ids = new nint[n];
            err = OCL.GetDeviceIDs(Handle, type, n, ids, (uint *)null);
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
            CLHandle.OCL.ReleaseDevice(Handle);
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

            int err;
            QueueProperties[] props = [
                (QueueProperties)CommandQueueInfo.Properties, (QueueProperties) CommandQueueProperties.ProfilingEnable,
                0
            ];
            fixed (QueueProperties *p = props)
            {
                Handle = OCL.CreateCommandQueueWithProperties(context.Handle, device.Handle, p, &err);
            }

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't create command queue, code: {err}");
            }
        }

        public void Finish()
        {
            int err = CLHandle.OCL.Finish(Handle);

            if (err != (int)ErrorCodes.Success)
            {
                throw new System.Exception($"Couldn't finish command queue, code: {err}");
            }
        }

        private static nint[]? Nintize(Event[]? evs)
        {
            return evs?.Select((ev, i) => ev.Handle).ToArray();
        }

        public unsafe void EnqueueNDRangeKernel(
            Kernel kernel,
            NDRange offset,
            NDRange global,
            NDRange local,
            out Event @event,
            Event[]? wait_list = null)
        {

            int err;
            nint event_h;
            fixed (nuint *g = global.Sizes)
            fixed (nuint *o = offset.Sizes)
            fixed (nuint *l = local.Sizes)
            fixed (nint *wait_list_p = Nintize(wait_list))
            {
                err = OCL.EnqueueNdrangeKernel(
                    Handle,
                    kernel.Handle,
                    global.Dimensions,
                    offset.Dimensions != 0 ? o : null,
                    g,
                    l,
                    wait_list == null ? 0 : (uint) wait_list.Length,
                    wait_list == null ? null : wait_list_p,
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
            nint event_h;
            var ptr = OCL.EnqueueMapBuffer(
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
            out Event @event,
            Event[]? wait_list = null)
        where T : unmanaged
        {

            nint event_h;
            int err;
            fixed (nint *wait_list_p = Nintize(wait_list))
            {
                err = OCL.EnqueueReadBuffer(
                    Handle,
                    buffer.Handle,
                    blocking,
                    offset,
                    array.Count * (nuint) sizeof(T),
                    array.Buf,
                    wait_list == null ? 0 : (uint) wait_list.Length,
                    wait_list == null ? null : wait_list_p,
                    out event_h);
            }
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

            nint event_h;
            int err = OCL.EnqueueWriteBuffer(
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

            nint event_h;
            int err = OCL.EnqueueUnmapMemObject(
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

        public unsafe void EnqueueCopyBuffer<T>(
            Buffer<T> src,
            Buffer<T> dst,
            nuint src_offset,
            nuint dst_offset,
            nuint count,
            out Event @event,
            Event[]? wait_list = null)
        where T : unmanaged
        {
            nint event_h;
            int err;
            fixed (nint *wait_list_p = Nintize(wait_list))
            {
                err = OCL.EnqueueCopyBuffer(
                    Handle,
                    src.Handle,
                    dst.Handle,
                    src_offset,
                    dst_offset,
                    count * (nuint) sizeof(T),
                    wait_list == null ? 0 : (uint) wait_list.Length,
                    wait_list == null ? null : wait_list_p,
                    out event_h);
            }

            if (err != (int)ErrorCodes.Success)
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
            CLHandle.OCL.ReleaseCommandQueue(Handle);
        }
    }

    class Kernel
    {
        public nint Handle { get; }

        unsafe public Kernel(
            Program program,
            string name)
        {
            int err;
            Handle = OCL.CreateKernel(program.Handle, Encoding.ASCII.GetBytes(name), &err);

            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create kernel, code: {err}");
            }
        }

        ~Kernel()
        {
            CLHandle.OCL.ReleaseKernel(Handle);
        }
        
        unsafe private void GetKernelInfo<Y>(
            uint arg_index,
            KernelArgInfo param_name,
            nuint param_value_size,
            Y *param_value,
            nuint *param_value_size_ret)
        where Y: unmanaged
        {
            int err = OCL.GetKernelArgInfo(
                Handle,
                arg_index,
                param_name,
                param_value_size,
                param_value,
                param_value_size_ret);
                
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception(
                    $"Failed to get kernel argument info (index = {arg_index}), code: {err}");
            }
        }
        
        unsafe public string GetArgTypeName(
            uint arg_index
        )
        {
            nuint size_ret;
            GetKernelInfo<byte>(
                arg_index,
                KernelArgInfo.TypeName, 
                0, null,
                &size_ret);

            byte[] infoBytes = new byte[size_ret / (nuint)sizeof(byte)];
            
            fixed (byte *p_infoBytes = infoBytes)
            {
                GetKernelInfo(
                    arg_index,
                    KernelArgInfo.TypeName, 
                    size_ret, p_infoBytes,
                    null);
            }

            return Encoding.UTF8.GetString(infoBytes);
        }
        
        unsafe public KernelArgAddressQualifier GetArgAddressQualifier(
            uint arg_index
        )
        {
            KernelArgAddressQualifier res;

            GetKernelInfo<KernelArgAddressQualifier>(
                arg_index,
                KernelArgInfo.AddressQualifier, 
                sizeof(KernelArgAddressQualifier), &res,
                null);

            return res;
        }

        unsafe public void SetArg<T>(
            uint arg_index,
            SparkOCL.Buffer<T> buffer)
        where T : unmanaged, INumber<T>
        {
            var binding = buffer.Handle;

            int err = OCL.SetKernelArg(Handle, arg_index, (nuint)sizeof(nint), ref binding);
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
            int err = OCL.SetKernelArg(Handle, arg_index, (nuint)sizeof(T), ref arg);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to set kernel argument, code: {err}, arg size: {(nuint)sizeof(T)}");
            }
        }

        unsafe public void SetSize(
            uint arg_index,
            nuint sz)
        {
            int err = OCL.SetKernelArg(Handle, arg_index, (nuint)sizeof(float) * sz, null);
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
            int err;
            Handle = OCL.CreateBuffer(context.Handle, flags, (nuint) sizeof(T) * array.Count, array.Buf, &err);
            if (err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create buffer, code: {err}");
            }
        }

        ~Buffer()
        {
            CLHandle.OCL.ReleaseMemObject(Handle);
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
            Count = (nuint)array.Length;
            array.CopyTo(new Span<T>(Buf, array.Length));
        }

        public Array (nuint size)
        {
            ElementSize = (nuint)sizeof(T);
            Buf = (T*)NativeMemory.AlignedAlloc(size * ElementSize, 4096);
            Count = size;
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
            using StreamReader sr = new StreamReader(fileName);
            string clStr = sr.ReadToEnd();

            int err;
            var program = OCL.CreateProgramWithSource(context.Handle, 1, new string[] { clStr }, null, &err);
            if (program == IntPtr.Zero || err != (int) ErrorCodes.Success)
            {
                throw new System.Exception($"Failed to create CL program from source, code: {err}");
            }

            var errNum = OCL.BuildProgram(program, 0, null, (byte*)null, null, null);

            if (errNum != (int)ErrorCodes.Success)
            {
                _ = OCL.GetProgramBuildInfo(program, device.Handle, ProgramBuildInfo.BuildLog, 0, null, out nuint buildLogSize);
                byte[] log = new byte[buildLogSize / (nuint)sizeof(byte)];
                fixed (void* pValue = log)
                {
                    OCL.GetProgramBuildInfo(program, device.Handle, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
                }
                string? build_log = System.Text.Encoding.UTF8.GetString(log);

                //Console.WriteLine("Error in kernel: ");
                Console.WriteLine("=============== OpenCL Program Build Info ================");
                Console.WriteLine(build_log);
                Console.WriteLine("==========================================================");

                OCL.ReleaseProgram(program);
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
            CLHandle.OCL.ReleaseProgram(Handle);
        }
    }
}
