# TreeEvo-GAN

Review-stage code with CIFAR-10 search and retraining examples. Additional
experiment-specific reproduction materials will be released upon publication.

Run on Linux with Python 3.8 and an NVIDIA CUDA GPU. From the repository root:

```bash
conda create -n treeevo python=3.8 -y
conda activate treeevo
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt

# Train the supernet and search
GPU=0 DATA_PATH=/path/to/cifar10 bash scripts/search_cifar10.sh

# Retrain the selected architecture
GPU=0 DATA_PATH=/path/to/cifar10 GENOTYPE=best_fid_0.npy bash scripts/retrain_cifar10.sh
```

`exps/best_fid_0.npy` is the selected main FID-preferred generator architecture;
`exps/best_fid_1.npy` is an additional FID-preferred architecture. Both have shape
`(3, 7)`. The `*_D.npy` files specify fixed discriminator genotypes.
Use the supplied CIFAR-10 FID reference in `fid_stat/`.

Based on [EWSGAN](https://github.com/weinnaTT/EWSGAN).
