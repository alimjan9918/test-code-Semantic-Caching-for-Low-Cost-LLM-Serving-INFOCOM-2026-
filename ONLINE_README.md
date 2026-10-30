# 图2(c)在线实验

在已有 fig2a_results 文件夹存在时运行：

    python experiment_online.py

默认 m=20、k=5、T=5000、10个种子，比较 CUCB-SC、CLCB-SC、CLCB-SC-LS、Epsilon-Greedy、LFU。输出 fig2c_results/fig2c_online.png 和 online.csv。

CLCB-SC-LS 使用算法3中的阶段触发条件；算法4用LCB和Reverse Greedy切换缓存；算法5比较LCB与最近缓存距离，决定调用模型或复用缓存。当前代码中的噪声、成本、模型和查询集来自图2(a)参考实验。实际运行没有调用LLM，填充成本使用固定c模拟。

论文未公开完整代码、查询集、日志协议、delta和精确alpha；图中默认显示相对精确最优缓存的平均额外成本，不能直接称为论文理论regret。LFU使用完整请求流频率，是静态事后基线。
