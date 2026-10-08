# 数据集准备

仓库附带 3 个 <1 MB 的小数据集，可以直接运行：`yale_16x16.bin`（N=165）、`fashion_16x16_N200.bin`、`mnist_16x16_N200.bin`。60k 全量二进制每个约 120 MB，不入库（已列入 `.gitignore`），按下面的方法生成。`*.meta.txt` 记录了每个文件的来源和参数。

所有脚本只依赖 Python 标准库，输出统一格式（little-endian）：

```
uint32 N            # 样本数
uint32 d            # 维度（16×16 = 256）
float64 × N × d     # row-major，像素 0..255，未归一化（C++ 端负责中心化）
```

## Yale Face Database（N=165）

1. 下载 Yale Face Database，解压到 `data/YALE/`，确保存在 `data/YALE/centered/subject*.pgm`
2. 生成：

```bash
python3 scripts/prepare_yale.py --target 16          # → data/yale_16x16.bin
```

## Fashion-MNIST / MNIST

1. 下载 IDX 文件：
   - Fashion-MNIST：<https://github.com/zalandoresearch/fashion-mnist/tree/master/data/fashion> → `data/fashion/`
   - MNIST：<https://ossci-datasets.s3.amazonaws.com/mnist/> → `data/mnist/`
2. 生成：

```bash
# 全量 60 000 训练集
python3 scripts/prepare_idx.py --src data/fashion/train-images-idx3-ubyte.gz \
    --target 16 --out data/fashion_16x16_full.bin
python3 scripts/prepare_idx.py --src data/mnist/train-images-idx3-ubyte.gz \
    --target 16 --out data/mnist_16x16_train_full.bin

# N=200 子集（docs/HE_THREE_DATASETS_REPORT.md 使用）
python3 scripts/prepare_idx.py --src data/fashion/train-images-idx3-ubyte.gz \
    --target 16 --limit 200 --out data/fashion_16x16_N200.bin
python3 scripts/prepare_idx.py --src data/mnist/t10k-images-idx3-ubyte.gz \
    --target 16 --limit 200 --out data/mnist_16x16_N200.bin
```

不设 `DATASET_FILE` 时，`he_pca` 会生成合成低秩数据（见 `TRUE_RANK`、`NOISE_SIGMA`），可以在没有数据集的情况下直接运行。
