# 图2(b)离线实验

将 experiment_offline.py 和 experiment_known_parameters.py 放在同一目录。保留已完成的 fig2a_results 文件夹。激活此前虚拟环境后运行：

    python experiment_offline.py

无需重新下载向量模型，不运行旧版 reproduce.py。
默认 m=20、k=5、10个随机种子，n=300,325,...,600，对应论文图2(b)展示范围。比较CUCB-SC、CLCB-SC、Epsilon-Greedy（删除时概率0.2）、LFU。

输入复用每个种子在图2(a)构造的固定真实成本、同一距离矩阵及真实均匀概率。策略只能访问日志与距离；评分器才读取真实成本和概率。

算法2的计数与UCB公式按截图实现。不额外裁剪UCB/LCB。零观测时置信界和经验估计存在未定义问题，显式约定UCB=1、LCB=0、经验值=1并记录零观测数。观测值0是有效成本，以NaN表示无反馈，不能用cost>0判断是否观测。

论文未给出日志采集策略。本实验显式采用：每条请求以nu概率使用空缓存并调用模型，否则使用全查询缓存复用答案；默认nu=0.5。日志采集缓存可以不同于目标容量5，全缓存只是构造部分反馈日志的一种明确约定，不是论文确认策略。所有算法共享日志；不同n使用同一日志前缀。

默认反馈直接等于已构造的固定成本c，故不再叠加第二次噪声；这不是已确认的作者观测机制。若要额外测试正文C_t=c+epsilon模型，可显式传入 --feedback-std 0.05。该额外噪声不截断，可能出界，因此统计性质与归一化实验段的关系需单独讨论。

--delta 默认0.05，论文未给出实验delta。
--alpha 默认1，图显示 loss - exact_optimum，不冒充已知论文曲率近似因子。all_runs.csv同时保存excess_loss与loss-alpha*optimum。若有明确理论alpha，可传入 --alpha。没有自动算曲率，尤其本数据min-max生成零成本，曲率边界可能退化。

输出fig2b_results中包含：
- fig2b_offline.png：均值和样本标准差阴影；
- all_runs.csv：每种子每n每算法结果与缓存编号；
- summary.csv：汇总；
- parameter_estimates.csv：频率、成本估计、观测数、UCB/LCB；
- seed_XX_log.npz：请求、反馈、采集动作及真实评分参数；
- config.json：全部设置和未明确项。

该程序按算法2选择缓存并模拟缓存答案的存在，不实际调用LLM生成回答；离线静态损失不加入填充成本。查询集、模型和日志协议与作者实验可能不同，禁止为使曲线相似而调整结果。
