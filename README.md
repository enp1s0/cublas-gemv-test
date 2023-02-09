# cublasDgemv performance test

## Build
```bash
make
```

## Run
```
./cublas-gemv.test
```


## Result
- on NVIDIA A100 40GB (SXM4)
  - Maximum : 367 GFlop/s (1.47 TB/s, bandwidth efficiency : 94%)

```
N,bandwidth_in_tbyteps,throughput_in_tflops
128,2.581721e-02,6.355006e-03
256,9.485716e-02,2.353046e-02
512,8.015587e-02,1.996099e-02
1024,7.624656e-01,1.902448e-01
2048,8.782761e-01,2.193548e-01
4096,1.320650e+00,3.300013e-01
8192,1.405707e+00,3.513409e-01
16384,1.449703e+00,3.623814e-01
32768,1.462433e+00,3.655860e-01
65536,1.468638e+00,3.671483e-01
```

## License
MIT
