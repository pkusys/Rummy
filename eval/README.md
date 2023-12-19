## 0. Figure 9
Run the following code to get results of Rummy and Lower bound (the minimal time of transmission and computation time). The last value is nprobe/ncentroid.
```bash
git checkout main
make
./overall_billion deep 2048 10 0.03125
./overall_billion sift 2048 10 0.03125
./overall_billion text 2048 10 0.03125
./overall_billion deep 8 10 8 0.002605
./overall_billion sift 8 10 8 0.00391
./overall_billion text 8 10 8 0.0053
```

Run the following code to get results of IVF-Rotation.
```bash
git checkout faiss-naive
make
./overall_billion deep 2048 10 16 0.03125
./overall_billion sift 2048 10 16 0.03125
./overall_billion text 2048 10 32 0.03125
./overall_billion deep 8 10 8 16 0.002605
./overall_billion sift 8 10 8 16 0.00391
./overall_billion text 8 10 8 32 0.0053
```

Run the following code to get results of IVF-CUM.
```bash
git checkout unified-memory
make
./overall_billion deep 2048 10 0.03125
./overall_billion sift 2048 10 0.03125
./overall_billion text 2048 10 0.03125
./overall_billion deep 8 10 8 0.002605
./overall_billion sift 8 10 8 0.00391
./overall_billion text 8 10 8 0.0053
```

## 0. Figure 10
The results of Rummy is the same with that in Figure 9.
Run the following code to get results of IVF-CPU.
```bash
git checkout cpu
make
./overall deep 2048 10 0.03125
./overall sift 2048 10 0.03125
./overall text 2048 10 0.03125
./overall deep 8 10 8 0.002605
./overall sift 8 10 8 0.00391
./overall text 8 10 8 0.0053
```

## 1. Figure 11
Run the following code to get results of Rummy.
The last value is nprobe.
```bash
git checkout main
make
./overall_billion sift 2048 10 32
./overall_billion sift 2048 10 28
./overall_billion sift 2048 10 24
./overall_billion sift 2048 10 20
./overall_billion sift 2048 10 16
./overall_billion sift 2048 10 12
./overall_billion sift 2048 10 10
./overall_billion sift 2048 10 8
./overall_billion sift 2048 10 6
./overall_billion sift 2048 10 5
./overall_billion sift 2048 10 4
./overall_billion sift 2048 10 3
./overall_billion sift 2048 10 2
./overall_billion sift 2048 10 1

./overall_billion sift 8 10 32
./overall_billion sift 8 10 28
./overall_billion sift 8 10 24
./overall_billion sift 8 10 20
./overall_billion sift 8 10 16
./overall_billion sift 8 10 12
./overall_billion sift 8 10 10
./overall_billion sift 8 10 8
./overall_billion sift 8 10 6
./overall_billion sift 8 10 5
./overall_billion sift 8 10 4
./overall_billion sift 8 10 3
./overall_billion sift 8 10 2
./overall_billion sift 8 10 1
```

Run the following code to get results of IVF-CPU.
The last value is nprobe.
```bash
git checkout main
make
./overall sift 2048 10 32
./overall sift 2048 10 28
./overall sift 2048 10 24
./overall sift 2048 10 20
./overall sift 2048 10 16
./overall sift 2048 10 12
./overall sift 2048 10 10
./overall sift 2048 10 8
./overall sift 2048 10 6
./overall sift 2048 10 5
./overall sift 2048 10 4
./overall sift 2048 10 3
./overall sift 2048 10 2
./overall sift 2048 10 1

./overall sift 8 10 32
./overall sift 8 10 28
./overall sift 8 10 24
./overall sift 8 10 20
./overall sift 8 10 16
./overall sift 8 10 12
./overall sift 8 10 10
./overall sift 8 10 8
./overall sift 8 10 6
./overall sift 8 10 5
./overall sift 8 10 4
./overall sift 8 10 3
./overall sift 8 10 2
./overall sift 8 10 1
```
