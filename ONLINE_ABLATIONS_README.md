# 在线消融实验：图 2(d)、2(e)、2(f)

在本文件所在目录打开 PowerShell。需要已有 `fig2a_results` 输入文件夹，并安装 `numpy`、`matplotlib`。

先做快速测试：

```powershell
python experiment_online_ablations.py --rounds 200 --seeds 2 --k-values 2 5 --m-values 5 10 --k-for-m 2 --out fig2def_check
```

测试成功后运行默认配置：

```powershell
python experiment_online_ablations.py
```

默认配置为 `T=5000`、10 个随机种子；图 2(d) 扫描 `k=2,3,5,7,10`，图 2(e) 扫描 `m=10,15,20` 且固定 `k=5`。运行时间取决于电脑性能。正式结果保存在 `fig2def_results`：

- `fig2d_by_k.png`：最终平均额外成本随缓存大小变化；
- `fig2e_by_m.png`：最终平均额外成本随查询数量变化；
- `fig2f_switches.png`：平均缓存切换次数；
- `fig2f_runtime.png`：每个随机种子的算法运行时间；
- `fig2d_summary.csv`、`fig2e_summary.csv`、`fig2f_summary.csv`：绘图数值；
- `fig2d_online.csv`、`fig2e_online.csv`：逐种子、逐时间点的原始轨迹；
- `config.json`：本次参数及复现口径。

图 2(e) 的输入约定：现有数据只有 20 条查询，因此 `m<20` 使用原有查询、距离和成本的前 m 项，并将均匀概率重新归一化。论文没有公布不同 m 对应的查询集和到达概率，此处是明确记录的参考设定，不保证数值与论文一致。`mean_runtime_seconds` 是本机测得值，不应与论文机器上的绝对时间直接比较。

