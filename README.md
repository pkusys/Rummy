## 0. Introduction
Rummy is a GPU-based vector query processing system for billion-scale datasets beyond GPU memory (with a single GPU). This repository contains one version of the source code. You can find more
details about the system design in our NSDI'24 paper: "" [[Paper]]().

## 1. Implementation and reproducing
The core implementation is in [DIR](https://github.com/Gold-Sea/Faiss-GPU/tree/main/faiss/pipe)
and some other files with name `*pipe*`.
The evaluation code and running commands is in `<repo>/eval`.
Please note that all running scripts are in the main branch.
You can refer to `<repo>/Setup.md` to set up the environment.

## 2. Contact
For any question, please contact `zzlcs at pku dot edu dot cn`.

The original documentation of Faiss is [here](https://github.com/Gold-Sea/Faiss-GPU/blob/main/README-faiss.md).
