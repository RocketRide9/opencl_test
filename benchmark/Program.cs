// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;
using Quasar.Native;
using Solvers;

var summaries = BenchmarkSwitcher.FromAssembly(typeof(TestCPU).Assembly).RunAll();

[SimpleJob(RuntimeMoniker.Net80)]
public class TestCPU
{
    float[] x;
    float[] y;
    float[] z;
    
    [Params(2000000, 8000000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        var rnd = new Random();
        x = new float[N];
        y = new float[N];
        z = new float[N];
    
        for(int i = 0; i < N; i++)
        {
            x[i] = rnd.NextSingle();
            y[i] = rnd.NextSingle();
        }
    }
    
    [Benchmark]
    public void MKL()
    {
        x.CopyTo(z, 0);
        BLAS.axpy(x.Length, -1, y, z);
    }

    [Benchmark (Baseline = true)]
    public void CSharp()
    {
        Shared.MyFor(0, x.Length, i =>
        {
            z[i] = x[i] - y[i];
        });
    }

}
