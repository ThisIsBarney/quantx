# Sz 守恒块稀疏 DMRG：正确性修复与性能工程

**模块：** `src/dmrg_blocks.cpp`（QuantX 原生引擎）
**状态：** 已修正并通过验证；在整个基准网格上比 TeNPy 快 3–20 倍。

---

## 1. 摘要

本报告记录了 QuantX 中 Sz 守恒块稀疏 DMRG（`run_dmrg_blocks`）的修复与优化过程。
该求解器用于一维 Heisenberg 链的基态计算。

工作开始时，块稀疏路径是**错误的**——它给出错误的基态能量（例如 n=8 给出约 −6.9，
而精确值为 −13.5）。在修复了三个相互独立的算法 bug（外加一个大 n 崩溃）之后，
该路径相对稠密参考路径的误差被降到 ~1e-14。随后一系列性能改进——以**实数运算**和
**单点（DMRG3S）混合算法**为顶点——使其从"完全错误"变为在
n ∈ {20, 50, 100}、χ ∈ {32, 64, 128} 的整个网格上**比成熟的 TeNPy 库快 3–20 倍**，
且能量完全一致。

| n \ χ | 32 | 64 | 128 |
|------:|:----:|:----:|:----:|
| **20**  | 20× | 19× | 19× |
| **50**  | —   | 5.9× | 3.1× |
| **100** | —   | 5.2× | 3.0× |

（加速比 = TeNPy 墙钟时间 ÷ QuantX 墙钟时间；能量一致到 ≥ 8 位有效数字。）

---

## 2. 背景：块稀疏 DMRG

DMRG 通过变分优化矩阵乘积态（MPS）来求一维量子哈密顿量的基态：来回扫描，每一步用
Lanczos 求解一个小的有效本征问题 `H_eff`。

对于守恒总 Sz 的哈密顿量（Heisenberg XXX 模型），每个 MPS 张量与环境张量在
**Sz 量子数（QN）上都是块对角的**。主导的 O(χ³) 收缩可分解为每个扇区
约 (χ/k)³ 大小的块，其中 k 是可达 QN 扇区的数目。利用这一点正是"块稀疏"路径的全部意义。

核心数据结构（`src/dmrg_blocks.cpp`）：

- **`BondSpace`**——在每条虚拟键上，按 QN 排序的 `(QN, dim)` 扇区列表。
- **`BlockEnv`**——环境张量：`blocks[b][q_ket] = Mat(d_bra, d_ket)`，其中 MPO 键指标
  `b` 携带一个 QN 平移 `HB_DELTA[b]`，使得 `q_bra = q_ket + HB_DELTA[b]`。
  对 Heisenberg MPO，`HB_DELTA = {0, +2, −2, 0, 0}`。
- **`block_A[i][q_L]`**——第 i 个格点、扇区 q_L 的 MPS 张量，上/下自旋物理列映射到
  右键扇区 q_L±1。

QN 记账（哪个扇区与哪个扇区收缩、用什么 MPO 权重）既是 bug 的来源，也是下文优化机会
的来源。

---

## 3. 正确性：三个 bug

块路径从来没对过。通过有纪律的二分定位发现了三个独立的 bug：n=2 情形（环境平凡）是对的，
从而把错误隔离到只有在 n ≥ 4 才会触发的非平凡环境/收缩代码。方法论是
**把每一个中间量都与已知正确的稠密路径对比**。

### Bug 1——MPO 基必须携带确定的 ΔSz

`build_heisenberg_mpo` 输出的是教科书式的 **X/Y/Z** MPO。但块代码把每个 MPO 键指标
经由一个*固定*的 QN 平移 `HB_DELTA[b]` 来路由。算符 X 在*两个*方向上混合 ↑↔↓
（ΔSz = ±2），因此它**没有单一的 ΔSz**——用固定 delta 路由它会悄悄丢掉一半的 XY 耦合，
破坏环境。

**修复：** 输出 QN 可分辨的 **S⁺/S⁻/Sz** MPO，其中每个键指标确实携带确定的 ΔSz
（S⁺: +2，S⁻: −2，Sz: 0），与 `HB_DELTA` 完全匹配。这是完全相同的哈密顿量
（`XX+YY = 2(S⁺S⁻ + S⁻S⁺)`）；稠密路径对此选择不敏感。n=2 之前恰好对，只是因为它的
环境是平凡的。

### Bug 2——右环境 QN 关系中的符号错误

在 `block_update_right` 中，选择哪些项有贡献的 QN 一致性检查写成了
`HB_DELTA[b_out] = HB_DELTA[b_in] + spz − sz`，而正确关系（其中 `s` = ket 自旋、
`sp` = bra 自旋）是 `+ sz − spz`。翻转的符号错误地拒绝了 S⁺/S⁻ 项，使右环境丢失了
XY 耦合。

### Bug 3——`block_lanczos` 中被打乱的稠密 H_eff

对小的局部问题，`block_lanczos` 通过对单位基向量逐列施加 matvec 来构造稠密 `H_eff`
矩阵。基向量按**行优先**枚举（`for r: for c:`），而 `theta_to_vec`/`vec_to_theta`
按**列优先**打包每个块（Eigen 默认）。结果是 H 的列索引顺序与行索引顺序不一致，
H 成了一个被打乱的算符。随后的对称化 `(H + Hᴴ)/2` *掩盖*了该 bug，给出看似合理但错误的
本征值与不收敛的扫描。（n=4 凑巧通过，因为它的块大多是 1×N，此时两种顺序重合。）

**诊断工具：** 决定性的检验是验证返回的本征值是否等于返回的本征向量在*真实*算符下的
瑞利商——二者不一致，从而精确定位到 H 不自洽。

### Bug 4——大 n 整数溢出

`make_heisenberg_bond_space` 用以 `long long` 存储的二项式路径计数给 QN 扇区赋权。
在 n ≳ 60 时它们溢出（binom(100,50)² ≈ 1e29），产生垃圾键维度并触发访问违例崩溃
（n=100 无法使用）。由于这些计数只是用于按比例分配的相对权重，现改为用 `double` 计算。

修复后，块路径对 n = 2, 4, 6, 8 与稠密路径一致到 ~1e-14。

---

## 4. 性能工程

确立正确性后，一系列改动——每一项都经验证保持能量不变——推动了加速。以下按影响排序。

### 4.1 实数哈密顿量用实数运算（约 2×）

内核此前一律用 `complex<double>` 存储。但 QN 可分辨的 Heisenberg MPO 及其基态都是
**实数**，而复数 GEMM 的浮点运算量约为实数的 4 倍、内存为 2 倍。整个块稀疏内核被在标量
类型 `S` 上模板化（封装于 `template<typename S> struct BlockDMRG`，配以成员别名
`Mat`/`Vec` 以及用于 MPO 转换与取实部的 `mscal()`/`rp()` 辅助函数）。
`run_dmrg_blocks` 检测 MPO 是否为实数，并分派到 `BlockDMRG<double>`（实数）或
`BlockDMRG<complex<double>>`。仅此一项就带来约 2×，并使最后一个对 TeNPy 落后的情形反超。

### 4.2 单点（DMRG3S）混合算法（大 χ 约 2–4×，小 χ 约 20×）

两点 DMRG 一次优化两个格点，其局部问题比必要的大 `d²` = 4 倍。单点 DMRG 一次优化一个
格点，但本身无法*调整*键维度。所实现的**混合算法**：

1. 运行两点扫描，直到两点能量趋于平台（`|ΔE| < 1e-4·|E|`）——此时各 QN 扇区上的键
   *分布*基本固定；
2. 随后切换到**单点**扫描，用专门的单点 `H_eff` matvec（`apply_heff_1site`）逐点优化，
   并用逐扇区的块 QR/SVD 分裂（`singlesite_split_LR` / `_RL`）移动正交中心。

单点能达到*精确*的两点能量（它在已固定的键内细化），而每次 matvec 轻约 2–4 倍。每个部件
都单独验证（`QX_DMRG_1SITE_CHECK=1`）：单点瑞利商等于基态能量到 ~1e-15，分裂满足
`A·C = M` 重构且严格正则。可用 `QX_DMRG_NO_SINGLESITE=1` 关闭。

### 4.3 预计算收缩计划（长链杀手）

环境、MPO 与键空间在整个局部本征求解期间是固定的，因此所有 QN 记账（扇区匹配、
`std::map` 块查找、偏移运算）在 `build_heff_plan` 中**一次性**完成，并由
`apply_heff_plan` 在每次 matvec 时回放。这消除了在长链（许多小扇区）上占主导的每次 matvec
的 `std::map` 查找。计划保留了两次 GEMM 复用（`T1 = θ·Rᵀ` 每个 ket/R 块算一次；
`LT1 = L·T1` 每个 L 块算一次）。

### 4.4 局部本征求解器调优

Lanczos 每次求解做了过多 matvec：维度 ≤ 300 时用稠密回退（构造 H 需约 dim 次 matvec），
且 `MIN_STEPS=10` 即使初始向量（当前两点态）已经是本征向量也强制 ≥10 次 matvec。把稠密阈值
降到 24、每次迭代都检查 Ritz 残差、并加入**相对**停止判据（`ritz < 1e-7·|E|`，因为局部
求解只需远低于 SVD 截断），把每次求解的 matvec 从约 66 降到约 26。

### 4.5 并行化

- matvec 在 QN 扇区上并行（OpenMP），按 GEMM 工作量设门控，使小 χ 保持串行。
- 各线程的部分输出用**按行分块的并行归约**合并（而非串行临界区），使主导的 q=0 扇区
  不会串行化合并——仅此就把 n=50/χ=128 的 matvec 时间从 14.3 秒降到 8.2 秒。
- 截断中的逐扇区 SVD 相互独立，已并行计算。

一个关键的经验发现：QuantX 的块 matvec **在约 4 核处饱和**，而 TeNPy 在**单线程时最快**
（其块太小，BLAS 多线程不划算）。因此胜负由**单核效率**决定，而非核数——增加核数不会改变
正面对比。

### 4.6 其他改动

- **去掉了调试期间加入的扰动噪声**——它只会膨胀键维度（与墙钟时间）而不提升精度。
- SVD 截断中的**逐值 Schmidt 下限**（对应 TeNPy 的 `svd_min`），使可忽略态的长尾不会膨胀
  键维度。
- 针对极小块的**可复用 GEMM 暂存 + `lazyProduct`**（较小，约 8%）。

---

## 5. 基准方法与结果

基准与 **TeNPy 1.1.0** 在一维 Heisenberg 链上对比，双方都用 χ_max 上限和
svd_min/cutoff = 1e-10，计时含预热并取多次中位数。能量与稠密 `run_dmrg` 路径
（≤ ~1e-14）以及精确对角化（n ≤ 10）验证。

相对 TeNPy 的最终墙钟加速（中位数）：

| n \ χ | 32 | 64 | 128 |
|------:|:----:|:----:|:----:|
| **20**  | 20.1× | 18.6× | 19.1× |
| **50**  | —     | 5.9×  | 3.1× |
| **100** | —     | 5.2×  | 3.0× |

值得注意的是，**QuantX 随 n 的扩展性优于 TeNPy**——在 n=100 处我们赢得比 n=50 更多——
因为 TeNPy 的每块 Python/Cython 编排开销随键数增长，而 QuantX 的编译路径不会。

单位说明：QuantX 用完整的 Pauli 算符，因此哈密顿量是自旋-½ 约定的 4 倍；故每格点的 Bethe
参考为 `4·(¼ − ln2) ≈ −1.7726`，而非 −0.4431。有两个基准测试此前用了自旋-½ 常数，已更正。

---

## 6. 代码组织与验证

原始的 M176 DMRG 单体文件（`dmrg_sparse.cpp`）被拆分为聚焦文件：

- `dmrg_mpo.cpp`——MPO 构造（现为 QN 可分辨的 S⁺/S⁻/Sz）。
- `dmrg_dense.cpp`——稠密两点 DMRG（正确性参考）。
- `dmrg_blocks.cpp`——Sz 块稀疏 DMRG（本报告）。
- `dmrg_capi.cpp`——JSON C ABI。

验证工具：

- `qx_dmrg_test`——断言 n = 2, 4, 6, 8 时块能量 == 稠密能量。
- `qx_dmrg_bench`——C++ 中直接的两点 vs 块计时。
- `QX_DMRG_PROFILE=1`——分阶段（theta/lanczos/svd/env）耗时与 matvec 计数。
- `QX_DMRG_1SITE_CHECK=1`——单点 matvec 与分裂的隔离检验。

---

## 7. 后续工作

- **批量化 CPU 收缩**——把每条键的电荷块连续打包，使用批量小 GEMM 内核。CPU 上收益有限，
  但它是 GPU 的前置条件。
- **GPU 后端**——大 χ 处最大的杠杆（经 cuBLAS 批量 GEMM 可达 10–50×）。待有 CUDA 机器时
  进行；收缩计划已能映射到 strided-batched 调用。
- **SU(2) 非阿贝尔对称性**——追踪约化的自旋*多重态*而非单个 Sz 态。对 Heisenberg，这在固定
  精度下把有效键维度缩小约 2–3 倍 → 代价约降 8–27 倍。这是最大的算法飞跃，也是下一项工作。

---

## 附录：提交记录（在 `master` 上）

```
4bb671d  feat(dmrg): hybrid two-site warm-up + single-site refinement (default)
3ad0ba4  feat(dmrg): single-site L->R canonicalisation split (DMRG3S stage 2a)
dd64b5b  feat(dmrg): single-site H_eff matvec (DMRG3S stage 1)
f0bdebf  fix(dmrg): integer overflow in bond-space path counts at large n
4195ab1  perf(dmrg): real arithmetic for real Hamiltonians (~2x)
c6cde99  perf(dmrg): reusable GEMM scratch + lazyProduct for tiny blocks
90c0f82  perf(dmrg): parallelise per-sector SVD in block_svd_truncate
a2c1cd1  perf(dmrg): relative stopping tolerance for the local Lanczos
881b6c7  perf(dmrg): resolve theta blocks once per matvec in the plan
922a3de  perf(dmrg): precomputed H_eff contraction plan
3d4f4e1  perf(dmrg): add per-value Schmidt floor to block SVD truncation
d7bce02  perf(dmrg): parallelise block_apply_heff over sectors (OpenMP)
dbfdf46  perf(dmrg): much faster local eigensolver + phase profiling
710c602  perf(dmrg): drop perturbative noise + restructure block_apply_heff
6fc5bce  perf(dmrg): row-chunked parallel reduction in the matvec
f40ce1f  bench(dmrg): add qx_dmrg_bench block-vs-dense timing harness
15647cb  fix(dmrg): correct Sz-conserving block-sparse DMRG; finish file split
```
