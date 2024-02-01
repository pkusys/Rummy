## 0. Figure 9
Run the following code to get results of Rummy and Lower bound (the minimal time of transmission and computation time). The fist param is dataset name. The second param is BS.
The third param is topk.
The last param is nprobe/ncentroid ($<1$) or nprobe($\geq 1$).
```bash
git checkout main
# recompile faiss following <repo>/Setup.md
make -C build -j overall_billion
./overall_billion deep 2048 10 0.03125
./overall_billion sift 2048 10 0.03125
./overall_billion text 2048 10 0.03125
./overall_billion deep 8 10 0.002605
./overall_billion sift 8 10 0.00391
./overall_billion text 8 10 0.0053
```

Run the following code to get results of IVF-Rotation.
```bash
git checkout faiss-naive
# recompile faiss following <repo>/Setup.md
make -C build -j overall_billion
./overall_billion deep 2048 10 0.03125
./overall_billion sift 2048 10 0.03125
./overall_billion text 2048 10 0.03125
./overall_billion deep 8 10 0.002605
./overall_billion sift 8 10 0.00391
./overall_billion text 8 10 0.0053
```

Run the following code to get results of IVF-CUM.
```bash
git checkout unified-memory
# recompile faiss following <repo>/Setup.md
make -C build -j overall_billion
./overall_billion deep 2048 10 0.03125
./overall_billion sift 2048 10 0.03125
./overall_billion text 2048 10 0.03125
./overall_billion deep 8 10 0.002605
./overall_billion sift 8 10 0.00391
./overall_billion text 8 10 0.0053
```

## 1. Figure 10
The results of Rummy is the same with that in Figure 9.
Run the following code to get results of IVF-CPU.
```bash
git checkout cpu
# recompile faiss following <repo>/Setup.md
make -C build -j overall
./overall deep 2048 10 0.03125
./overall sift 2048 10 0.03125
./overall text 2048 10 0.03125
./overall deep 8 10 0.002605
./overall sift 8 10 0.00391
./overall text 8 10 0.0053
```

## 2. Figure 11
Run the following code to get results of Rummy.
The last param is nprobe.
```bash
git checkout main
# recompile faiss following <repo>/Setup.md
make -C build -j overall_billion
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
The last param is nprobe.
```bash
git checkout cpu
# recompile faiss following <repo>/Setup.md
make -C build -j overall
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

## 3. Figure 12
The last param means whether running original Faiss or Rummy (0 is Rummy).
```bash
git checkout comp
# recompile faiss following <repo>/Setup.md
make -C build -j comp
./comp 256 10 0.00390625 1
./comp 256 10 0.0078125 1
./comp 256 10 0.015625 1
./comp 256 10 0.03125 1
./comp 256 10 0.0625 1
./comp 256 10 0.125 1
./comp 256 10 0.25 1

./comp 256 10 0.00390625 0
./comp 256 10 0.0078125 0
./comp 256 10 0.015625 0
./comp 256 10 0.03125 0
./comp 256 10 0.0625 0
./comp 256 10 0.125 0
./comp 256 10 0.25 0

./comp 8 10 0.00390625 1
./comp 8 10 0.0078125 1
./comp 8 10 0.015625 1
./comp 8 10 0.03125 1
./comp 8 10 0.0625 1
./comp 8 10 0.125 1
./comp 8 10 0.25 1

./comp 8 10 0.00390625 0
./comp 8 10 0.0078125 0
./comp 8 10 0.015625 0
./comp 8 10 0.03125 0
./comp 8 10 0.0625 0
./comp 8 10 0.125 0
./comp 8 10 0.25 0
```

## 4. Figure 13
Since the profile points are numerous. The results are recorded in `profile_path`, and
the specific file name can be found in the `<repo>/eval/overall.cu` of branch **profile-eval**.
```bash
git checkout profile-eval
# recompile faiss following <repo>/Setup.md
make -C build -j overall
./overall deep 256 10
./overall sift 256 10
./overall text 256 10

./overall deep 8 10
./overall sift 8 10
./overall text 8 10
```

## 5. Figure 14
```bash
git checkout main
# recompile faiss following <repo>/Setup.md
make -C build -j overall
make -C build -j overall_no_reorder
make -C build -j overall_query
make -C build -j overall_per_cluster_pipeline
make -C build -j overall_no_pipeline

# Running Rummy and Lower Bound under large BS
./overall deep 256 10
./overall sift 256 10
./overall text30 256 10

# Running Rummy and Lower Bound under small BS
./overall deep 8 10
./overall sift 8 10
./overall text30 8 10

# Running No retrofitting under large BS
./overall_query deep 256 10
./overall_query sift 256 10
./overall_query text30 256 10

# Running No retrofitting under small BS
./overall_query deep 8 10
./overall_query sift 8 10
./overall_query text30 8 10

# Running No reorder under large BS
./overall_no_reorder.cu deep 256 10
./overall_no_reorder.cu sift 256 10
./overall_no_reorder.cu text30 256 10

# Running No reorder under small BS
./overall_no_reorder.cu deep 8 10
./overall_no_reorder.cu sift 8 10
./overall_no_reorder.cu text30 8 10

# Running per-cluster pipeline under large BS
./overall_per_cluster_pipeline deep 256 10
./overall_per_cluster_pipeline sift 256 10
./overall_per_cluster_pipeline text30 256 10

# Running per-cluster pipeline under small BS
./overall_per_cluster_pipeline deep 8 10
./overall_per_cluster_pipeline sift 8 10
./overall_per_cluster_pipeline text30 8 10

# Running one-group pipeline under large BS
./overall_no_pipeline deep 256 10
./overall_no_pipeline sift 256 10
./overall_no_pipeline text30 256 10

# Running one-group pipeline under small BS
./overall_no_pipeline deep 256 10
./overall_no_pipeline sift 256 10
./overall_no_pipeline text30 256 10
```

## 6. Figure 15
```bash
git checkout main
# recompile faiss following <repo>/Setup.md
make -C build -j overall
make -C build -j overall_no_lru
make -C build -j overall_no_pin
make -C build -j nomemman

# Running Rummy and Lower Bound under large BS
./overall deep 256 10
./overall sift 256 10
./overall text30 256 10

# Running Rummy and Lower Bound under small BS
./overall deep 8 10
./overall sift 8 10
./overall text30 8 10

# Running no replacement policy under large BS
./overall_no_lru deep 256 10
./overall_no_lru sift 256 10
./overall_no_lru text30 256 10

# Running no replacement policy under small BS
./overall_no_lru deep 8 10
./overall_no_lru sift 8 10
./overall_no_lru text30 8 10

# Running no pin memory under large BS
./overall_no_pin deep 256 10
./overall_no_pin sift 256 10
./overall_no_pin text30 256 10

# Running no pin memory under small BS
./overall_no_pin deep 8 10
./overall_no_pin sift 8 10
./overall_no_pin text30 8 10

# Running no memory management under large BS
./nomemman deep 256 10
./nomemman sift 256 10
./nomemman text30 256 10

# Running no memory management under small BS
./nomemman deep 8 10
./nomemman sift 8 10
./nomemman text30 8 10
```

## 7. Table 3
You can find the data in the logs when running Rummy in **Figure 14** or **Figure 15**.