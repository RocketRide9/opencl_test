// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

var summaries = BenchmarkSwitcher.FromAssembly(typeof(TestCPU).Assembly).RunAll();
[SimpleJob(RuntimeMoniker.Net80, baseline: true)]
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
    public double DotList()=>CPU_TEST.CPU.Dot(xl, yl);
    [Benchmark (Baseline =true)]
    public double DotArray()
    {
        double acc = 0;

        for (int i = 0; i < x.Length; i++)
        {
            acc += x[i] * y[i];
        }
        return acc;
    }

}
