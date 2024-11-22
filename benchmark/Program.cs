// See https://aka.ms/new-console-template for more information
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Running;

var summaries = BenchmarkSwitcher.FromAssembly(typeof(TestCPU).Assembly).RunAll();

[SimpleJob(RuntimeMoniker.Net80)]
public class TestCPU
{
    Solvers.MKL.BiCGStab mkl;
    Solvers.OpenCL.BiCGStab ocl;

    [GlobalSetup]
    public void Setup()
    {
        SparkCL.Core.Init();
        Console.WriteLine(Directory.GetCurrentDirectory());

        mkl = new Solvers.MKL.BiCGStab();
        ocl = new Solvers.OpenCL.BiCGStab();
    }
    [Benchmark]
    public (float, float, int) SolveMKL() => mkl.Solve();

    [Benchmark (Baseline =true)]
    public (float, float, int, long) SolveOCL() => ocl.Solve();

}
