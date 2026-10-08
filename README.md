# Lanczos-HE-PCA

基于 CKKS 全同态加密的**外包式隐私保护 PCA**：数据所有者（Client）把协方差矩阵加密后交给云端（Server），Server 在密文上执行完整的 Lanczos 迭代（含密态 Newton 1/√x 归一化、密态全重正交化 FRO 与 Bootstrapping），Client 只在最后解密一次，完成三对角特征分解与特征向量重构。

- 同态加密库：OpenFHE v1.5.1（CKKS，FLEXIBLEAUTO）
- 线性代数：Eigen 3.4.0
- 语言 / 构建：C++17，CMake ≥ 3.16（FetchContent 自动拉取依赖）

> 姊妹项目 **SFK-PCA**（多方联邦 + 随机特征核 PCA + 多方密钥 CKKS）在本项目的密态算子基础上做了扩展。

---

## 1. 技术方案

### 1.1 角色与信息流（2 轮交互）

| 角色 | 持有 | 职责 |
|---|---|---|
| Client（数据所有者） | 私钥、公钥 | 计算 C 并归一化 Ĉ = C / tr(C)；加密 Ĉ 与初始向量 v₀；最后解密 α、β、V_all，构造三对角 T_m，做 CW 过滤、特征分解和 Ritz 向量重构 |
| Server（云端） | 公钥、乘法/旋转/Bootstrap 密钥 | 在密文上执行 m 步 Lanczos：matvec、三项递推、内积、Newton 1/√x 归一化、FRO 重正交、Bootstrapping |

```
Client ── Enc(Ĉ), Enc(v₀), 公钥 ──▶ Server
                                    for j = 1..m:  W = Ĉ·V_j − β_{j−1}·V_{j−1}
                                                   α_j = ⟨V_j, W⟩;  W −= α_j·V_j;  (FRO)
                                                   β_j = ‖W‖;  V_{j+1} = W · Newton(‖W‖²)
Client ◀── Enc(α[]), Enc(β[]), Enc(V_all) ──
解密 → T_m → Cullum–Willoughby 过滤 → eig(T_m) → U_K = V_all · S_K
```

Server 全程只接触密文与公钥。

### 1.2 关键技术点

| 模块 | 做法 | 位置 |
|---|---|---|
| 矩阵-向量乘 | Halevi–Shoup **对角线打包 + BSGS**：d=256 时每次 matvec 的旋转次数从 3584 降到 32 | `src/server.cpp` |
| 向量归一化 | 密态 Newton 1/√x；每步初值由明文 mirror 预测 ‖W‖² 的范围；可选 aSOR 松弛加速 | `src/newton_inv_sqrt.*`, `src/asor.h` |
| 正交性保持 | 密态全重正交化（FRO），前 2 步跳过以节省深度 | `src/server.cpp` |
| 深度管理 | OpenFHE CKKS Bootstrapping（d=256、m=12 时约 24 次） | `src/server.cpp` |
| 鬼影过滤 | Cullum–Willoughby 过滤伪 Ritz 值 | `src/client.cpp` |
| 误差归因 | 同算术路径的明文 mirror，把误差拆成"迭代极限"和"FHE 增量" | `src/main.cpp` (`mirror` 模式) |

完整推导与参数设计见 [`docs/TECHNICAL_DOC.md`](docs/TECHNICAL_DOC.md)；用 d=4 例子手算整条流水线的走读见 [`docs/ALGORITHM_WALKTHROUGH.md`](docs/ALGORITHM_WALKTHROUGH.md)。

---

## 2. 构建与运行

### 2.1 依赖

- CMake ≥ 3.16、支持 C++17 的编译器
- macOS：`brew install libomp`
- 首次构建会通过 FetchContent 下载并编译 OpenFHE v1.5.1（约 10–20 分钟）

### 2.2 构建

```bash
./build.sh            # 等价于 cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

产物：`build/he_pca`（主程序）、`build/plaintext_lanczos_tune`（明文调参）、`build/depth_probe`（深度预算探针）。

### 2.3 准备数据

仓库已附带 Yale（N=165）以及 Fashion-MNIST / MNIST 的 N=200 子集，可以直接运行。60k 全量数据需要下载原始文件后生成，见 [`data/README.md`](data/README.md)。

### 2.4 运行

```bash
# 1) 最小 HE 冒烟测试（d=16，m=2，无 Bootstrap，数秒）
#    用来检查 HE 与明文 mirror 是否一致（FHE 增量约 1e-12）。m=2 步的 Krylov 子空间
#    本身不足以收敛，所以这里的特征值 / cos 精度很低属于预期。
./build/he_pca baseline

# 2) 合成低秩数据（d=64，m=8，K=3）
K=3 ./build/he_pca bootstrap 8 64

# 3) 真实数据：Yale d=256，K=4，m=12，Newton=4
DATASET_FILE=data/yale_16x16.bin K=4 NEWTON_ITERS=4 \
  ./build/he_pca bootstrap 12 256

# 4) 明文 mirror（秒级，用于和 HE 日志对照、调参）
DATASET_FILE=data/yale_16x16.bin K=4 ./build/he_pca mirror 12 256
```

常用环境变量：`DATASET_FILE`、`K`、`NEWTON_ITERS`、`ENABLE_FRO`（默认 1）、`FRO_SKIP_FIRST`（默认 2）、`BSGS_B`、`VIZ_DUMP_PATH`（导出重建图像）。批量脚本：`scripts/run_dimension_sweep.sh`、`scripts/run_mnist_fashion_full.sh`。

---

## 3. 实验结果

运行环境：Apple M1（48 GB RAM），单线程；RingDim 2¹⁷、numSlots 16384、depth 52、scale 59/60 bit。所有原始日志都在 `experiments/logs/`。

### 3.1 三数据集（d=256，K=4，m=8，Newton=3，N≈200 子集）

| 数据集 (N) | R²(X) 差距 (HE vs 明文) | R²(V) | Server 耗时 | Bootstrap 次数 |
|---|---|---|---|---|
| Yale (165) | **0.27 %** | 0.861 | 7.7 min | 16 |
| Fashion-MNIST (200) | 2.83 % | 0.772 | 7.5 min | 16 |
| MNIST (200) | 2.44 % | 0.555 | 7.7 min | 16 |

Fashion/MNIST 的差距主要来自 m=8 步 Krylov 子空间本身的极限（mirror 已验证），不是 HE 引入的。详见 [`docs/HE_THREE_DATASETS_REPORT.md`](docs/HE_THREE_DATASETS_REPORT.md)。

### 3.2 最新：全量数据（d=256，K=4，m=12，Newton=4）

| 数据集 | cos₁ | cos₂ | cos₃ | cos₄ | λ₁ / λ₂ 相对误差 | R²(X) 差距 | Server 耗时 |
|---|---|---|---|---|---|---|---|
| Yale (165) | 0.999999 | 0.99988 | 0.992 | 0.715 | 0.14 % / 0.18 % | 0.0029 | 18.6 min |
| Fashion-MNIST (60 000) | 0.99999 | 0.99999 | 0.984 | 0.451 | 0.16 % / 0.01 % | 0.0147 | 6.0 min |
| MNIST (60 000) | 0.99999 | 0.996 | 0.130 | 0.227 | 0.12 % / 0.18 % | 0.0331 | 6.2 min |

误差归因（来自日志里的"HE − mirror"拆分）：前两个主成分的误差几乎全部是 FHE 增量，量级只有 10⁻³–10⁻⁶。第 3、4 个主成分的偏差也以 FHE 增量为主：同一配置下明文 mirror 的 cos₃、cos₄ 都 ≥ 0.987（Yale cos₄ = 0.988，MNIST cos₄ = 0.998，Fashion 均 ≥ 0.99999）。所以低位主成分的精度瓶颈在密态数值（CKKS 噪声 + Newton 近似），而不在 Krylov 子空间。重建图像对比见 `experiments/viz/*/grid_he.png`，MSE 表见 `experiments/viz/*/mse_table.txt`。

### 3.3 性能优化（BSGS 对角线打包）

d=256 时，单次 matvec 的密文旋转从 3584 次降到 32 次；单次实验从约 37 min（行打包 + Newton-2）降到约 7.7 min（BSGS + Newton-3），内存降低约 60%。见 [`docs/WORK_REPORT.md`](docs/WORK_REPORT.md)。

---

## 4. 当前进展与已知限制

**已完成**

- 完整的密态 Lanczos-PCA 流水线：特征值与特征向量，2 轮交互
- 对角线打包 + BSGS matvec、密态 FRO、Bootstrapping、CW 过滤
- 明文 mirror 误差归因框架
- 在 Yale / Fashion-MNIST / MNIST（d=256，最多 60 000 样本）上的端到端实测

**已知限制**

1. **安全参数**：实验使用 `HEStd_NotSet` 手动设置 RingDim（2¹⁵ / 2¹⁷），尚未统一到 `HEStd_128_classic` 标准参数。
2. **低位主成分精度**：K=4 时第 3、4 个主成分的 FHE 增量误差偏大（见 3.2 节），需要更精确的归一化或更多 Newton 步。
3. **威胁模型**：单数据所有者 + 半诚实云端的外包场景；Client 最终会解密 α、β 和全部 Lanczos 基 V_all。多方联邦与输出最小化在姊妹项目 SFK-PCA 中处理。
4. **初值预测**：Newton 每步的初值依赖 Client 侧明文 mirror 对 ‖W‖² 范围的预测（`GUESS_SAFETY`），属于工程辅助，不属于协议本身。

---

## 5. 目录结构

```
.
├── CMakeLists.txt / build.sh      # 构建（FetchContent 拉取 OpenFHE 1.5.1 与 Eigen 3.4.0）
├── src/
│   ├── main.cpp                   # 端到端流程、计时、误差报告、mirror 模式
│   ├── client.{h,cpp}             # 归一化、加密、解密、T_m 构造、CW 过滤、特征向量重构
│   ├── server.{h,cpp}             # 密态 Lanczos：BSGS 对角 matvec、内积、FRO、Bootstrap
│   ├── newton_inv_sqrt.{h,cpp}    # 密态 Newton 1/√x（含 aSOR 版本）
│   ├── asor.h                     # aSOR 松弛因子预计算
│   ├── pca_eval.h                 # R²(X) / R²(V) / cos 等评价指标
│   ├── plaintext_lanczos_tune.cpp # 明文 Lanczos 调参工具
│   └── depth_probe.cpp            # CKKS 深度 / RingDim 探针
├── scripts/                       # 数据准备、批量实验、日志汇总、可视化
├── experiments/                   # logs/（原始日志）、results/、viz/（重建图与 MSE 表）
├── data/                          # 3 个小数据集 + 元信息 + 全量数据生成说明
└── docs/
    ├── TECHNICAL_DOC.md           # 技术文档（架构、参数、算法、打包、Bootstrap）
    ├── ALGORITHM_WALKTHROUGH.md   # d=4 全程手算走读
    ├── EVALUATION_METHODOLOGY.md  # 评价指标与实验方法学
    ├── WORK_REPORT.md             # BSGS 优化与多数据集实测报告
    ├── HE_THREE_DATASETS_REPORT.md
    ├── EXPERIMENT_REPORT.md       # 早期维度扫描总结（d=16…1024）
    ├── slides/                    # 方案两页 PPT
    └── history/                   # PoC 方案、方向2 改造方案、aSOR 方案等历史设计文档
```

## 6. 参考文献

- Panda, *Principal Component Analysis Using CKKS Homomorphic Scheme*, 2021（IACR ePrint 2021/914）
- Ma, Ma, Jiang, Ge, *Improved privacy-preserving PCA using optimized homomorphic matrix multiplication*, Computers & Security 138 (2024) 103658
- Moon et al., *Adaptive Successive Over-Relaxation Method for a Faster Iterative Approximation of Homomorphic Operations*, 2024（IACR ePrint 2024/1366）
- Halevi & Shoup, *Algorithms in HElib*, CRYPTO 2014（对角线打包）
- Cullum & Willoughby, *Lanczos Algorithms for Large Symmetric Eigenvalue Computations*
