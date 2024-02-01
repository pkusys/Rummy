# AWS Set Up

Please choose **AMI** ami-08cb7e65c4e13f22d or any AMI with cuda and cublas.

x1.16xlarge is for CPU instance. p4d.24xlarge, p3.2xlarge and  g4dn.2xlarg are for GPU instances.

## Docker Image

1. Just run command (docker is preinstalled) : `docker pull goldensea/faiss-gpu:v2`
2. Run `docker images` to check if the docker image is installed
3. map workspace and cuda into container

```bash
# for micro-benchmarks
docker run -it -v /home/ubuntu/workspace/:/workspace -v /usr/local/cuda:/usr/local/cuda/ --rm --network=host --ipc=host --gpus all goldensea/faiss-gpu:v2 bash

# for macro-benchmarks
docker run -it -v /home/ubuntu/workspace/:/workspace -v /usr/local/cuda:/usr/local/cuda/ -v /billion-data:/billion-data --rm --network=host --ipc=host --gpus all goldensea/faiss-gpu:v2 bash

# for cpu experiments
docker run -it -v /home/ubuntu/workspace/:/workspace -v /billion-data:/billion-data --rm --network=host --ipc=host goldensea/faiss-gpu:v2 bash
```

## AWS CLI

1. First Run the following commands to install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

1. Run `aws configure` to login in the AWS IAM, contact zzl to obtain the login key
2. Run `aws configure ls` to check the login status  

## Datasets

1.  `mkdir data-gpu && mkdir data` to create corresponding diectories.
2. Run the following commands to get the datasets from AWS S3. Since maintaining TBs-level datasets on AWS S3 is quite a burden for me. You are supposed to download the datasets on their official websites ([SIFT](http://corpus-texmex.irisa.fr/), [DEEP](https://research.yandex.com/datasets/biganns) and [TEXT](https://research.yandex.com/blog/benchmarks-for-billion-scale-similarity-search)).

```bash
# aws s3 cp --recursive s3://faiss-dataset/data-gpu/ ./data-gpu
# aws s3 cp --recursive s3://faiss-dataset/data/ ./data

cd /billion-data/data1
sudo aws s3 cp s3://faiss-dataset/billion-scale/deep1Bgtd.fvecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/deep1Bgti.ivecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/deep1B.fbin ./

cd /billion-data/data2
sudo aws s3 cp s3://faiss-dataset/billion-scale/sift1Bgtd.fvecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/sift1Bgti.ivecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/sift_learn.fvecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/sift1B.fbin ./

cd /billion-data/data3
sudo aws s3 cp s3://faiss-dataset/billion-scale/text1Bgti.ivecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/text1Bgtd.fvecs ./
sudo aws s3 cp s3://faiss-dataset/billion-scale/text1B.fbin ./

cd /billion-data/data4
sudo mkdir deep
sudo mkdir sift
sudo mkdir text
sudo aws s3 cp --recursive s3://faiss-dataset/data/deep/ ./deep
sudo aws s3 cp --recursive s3://faiss-dataset/data/sift/ ./sift
sudo aws s3 cp --recursive s3://faiss-dataset/data/text/ ./text
```

## Compile

1. Please reference to https://developer.nvidia.com/cuda-gpus to find the current GPU Compute Capability and fill in the number in `cmake -B build . -DBLA_VENDOR=Intel10_64_dyn -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_CUDA_ARCHITECTURES="$number”`

```
1. A100 : 80
2. V100 : 70
3. T4 : 75
```

Compile cpu version: `cmake -B build . -DBLA_VENDOR=Intel10_64_dyn -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF`

2. Run `make -C build -j faiss`.

3. RUN `make -C build -j overall`.

## Mount Local Block
I recommend you to use local disk in the instance since there may be some issues of EBS when facing large files.
```bash
sudo mkfs -t xfs /dev/nvme2n1
sudo mkfs -t xfs /dev/nvme3n1
sudo mkfs -t xfs /dev/nvme4n1
sudo mkfs -t xfs /dev/nvme5n1

sudo mount /dev/nvme2n1 /billion-data/data1
sudo mount /dev/nvme3n1 /billion-data/data2
sudo mount /dev/nvme4n1 /billion-data/data3
sudo mount /dev/nvme5n1 /billion-data/data4
```

***You can use spot instance to save money, but all of our experiments are conducted on normal instances.***