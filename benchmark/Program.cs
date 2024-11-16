// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

var summaries = BenchmarkSwitcher.FromAssembly(typeof(TestCPU).Assembly).RunAll();
[SimpleJob(RuntimeMoniker.HostProcess, baseline: true)]
[SimpleJob(RuntimeMoniker.Net90)]
public class TestCPU
{
    double[] x, y;
    List<double> xl, yl;
    [Params(10000, 100000, 2000000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        var rnd = new Random();
        x=new double[N];
        y=new double[N];
        
        for(int i = 0; i < N; i++) {
            x[i] = rnd.NextDouble();
            y[i]= rnd.NextDouble();}
        xl = new List<double>(x);
        yl = new List<double>(y);
    }
    [Benchmark]
    public double DotArray()=>CPU_TEST.CPU.Dot(x, y);
    [Benchmark]
    public double DotMKL()=> Quasar.Native.BLAS.dot(x.Length, x, y);

    [Benchmark (Baseline =true)]
    public double DotList()
    {
        double acc = 0;

        for (int i = 0; i < xl.Count; i++)
        {
            acc += xl[i] * yl[i];
        }
        return acc;
    }

}
