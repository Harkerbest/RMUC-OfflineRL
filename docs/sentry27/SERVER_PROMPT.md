# 给服务器上 agent 的提示词（v2）

第一轮（v1）的提示词和结果见 git 历史（`a3ced70`）。这一轮用 v2：新增偷基地/侧路目标、目标持续、自适应熵。

**开跑前你需要定一件事：`RULES`。** 先查 2027 规则手册里"基地控制点"的占领条件：

- 如果机器人站在敌方基地增益卡上就能占领敌方基地控制点，填 `arena`；
- 如果不行，填 `conservative`。

两种模式的区别见 `docs/sentry27/README.md` 第 10 节。把下面提示词里的 `<RULES>` 换成你的选择，再整段复制给服务器上的 agent。

```text
你在 Linux 服务器上（4×RTX 5880 Ada、144 核、128 GB 内存）继续训练 RMUC 2027 哨兵的决策层。
上一轮（v1）训练了 300 次更新，策略和行为树打平；这一轮用 v2 重新训练。
动手前先读 docs/sentry27/README.md，尤其是第 10、11 节（v1 的诊断和 v2 的改动）。

本轮规则：RULES=<RULES>

## 准备
1. 更新仓库（之前已经克隆过；没有的话按 README 克隆）：
   cd RMUC-OfflineRL && git checkout claude/inspiring-lamport-zd97ge && git pull
2. Arena-1 保持在 bbaa885，已编译的引擎可以直接用；设置环境变量：
   export ARENA1_ROOT=$(realpath ../Arena-1)
   不要修改 Arena-1，也不要把它复制进本仓库（非商业学习许可）。
3. 自检：python -m rm_rl.arena27.selftest
   必须输出 ALL PASS，其中 probe 一行应为 obs=258 g=342。有失败就停下，把输出发给我。

## 运行
用 tmux 或 nohup 运行，日志写到文件里。每一步都可以中断后用同一条命令续跑。
1. 基线评估，约 1 小时：
   RULES=<RULES> STAGE=baseline bash scripts/sentry27.sh
   先把 rm_runs/sentry27_v2_<RULES>_eval/summary.md 发给我。我要确认哪棵行为树最强。
   arena 模式的参照是 bt:raid，conservative 是 bt:balanced。
   如果 conservative 下 bt:assault 明显强于 bt:balanced（得分差的置信区间下界 > 0），告诉我，参照要改。
2. 训练，默认 300 次更新，约 2 天：
   RULES=<RULES> STAGE=train bash scripts/sentry27.sh
3. 最终评估，同时评估 latest 和每 50 次更新的检查点：
   RULES=<RULES> STAGE=eval bash scripts/sentry27.sh
4. 出图（不要覆盖 v1 的图）：
   python -m viz.sentry27_figures --train rm_runs/sentry27_v2_<RULES> --eval rm_runs/sentry27_v2_<RULES>_eval \
       --ref <arena 填 bt:raid，conservative 填 bt:balanced> --out viz/figures/sentry27_v2_<RULES>

## 资源
- 训练 worker 数默认为 CPU 数减 8。开跑后前 10 分钟用 free -g 观察：每个 worker 约 0.5 GB，
  内存超过 110 GB 就停掉，用 TRAIN_WORKERS=112 重启（会续跑）。
- 学习器自动用第一张 GPU，其余 GPU 用不上。瓶颈是 CPU。
- 停止训练：只给主进程（python -m rm_rl.train.train_sentry27）发 SIGTERM，它会一并关闭 worker。
  不要用 pkill -f 匹配宽泛的命令行，可能误杀别的进程，包括你自己的 shell。
  停止后检查没有残留的 spawn_main 或 rmuc-sim 进程。

## 训练中每隔几小时查看 rm_runs/sentry27_v2_<RULES>/train_log.csv
- 模仿阶段（前 6 次更新）结束时，imit_acc 应 ≥ 0.9。
  arena 模式模仿的是 raid，seat_enemy_base 应明显大于 0；conservative 模仿的是 balanced，seat_enemy_base 应为 0。
- PPO 阶段：
  - entropy 应在 0.5 附近（自适应熵的目标）。ent_coef 会自动调整：
    它长期顶在 0.05 而熵仍远低于 0.5，或者熵远高于 1.5，都要告诉我；
  - approx_kl 多数 < 0.03，clipfrac < 0.3，explained_var 应升到 0.5 以上；
  - 和 v1 对照：v1 的熵是 0.06，行为指标 300 次更新完全不变。
    这一轮如果 100 次更新后 seat_* 行为指标仍然一点不动，也告诉我。
- errors 应保持为 0。偶发几次可以；持续增长就停下，看日志里的 "rollout error"。
- 关注 seat_no_ammo、seat_own_central、seat_enemy_base、seat_deaths。行为出现异常时
  （例如得分上升但一直站着不动，或长时间完全不开火），去 results 里查那几局，判断是不是在利用仿真漏洞，并记录下来。
- 想在训练中途看学习曲线，可以少开 24 个训练 worker，用这 24 个核评估检查点：
  python -m rm_rl.eval.eval_sentry27 --out rm_runs/sentry27_v2_<RULES>_eval --workers 24 --seeds 1 \
      --ref <参照> [conservative 加 --no-base-raids] --arm <参照> --arm rl_0050=rm_runs/sentry27_v2_<RULES>/iter_0050
  也可以等训练结束后统一评估。

## 不要做
- 训练中途不要改观测、动作、奖励、对手组成或 RULES。这会破坏续跑和可比性。
  确实要改，就用新的 OUT 目录从头训练，并在汇报里写明改了什么、为什么。
- 不要修改 Arena-1，也不要为了让数字好看去改评估网格、参照对象或留出的队伍。
- 不要提交 rm_runs/ 下的 versions/、league/、state.pt、prior.pt。
- 不要用这一轮的代码评估 v1 的 rm_runs/sentry27/（会直接报错，这是有意的）。

## 结束后汇报给我
1. rm_runs/sentry27_v2_<RULES>_eval/summary.md 全文，加一句话结论：
   rl 相对参照行为树的配对得分差及 95% 置信区间，下界是否 > 0；rl 相对 script 是多少。
2. 学习曲线：各检查点 rl_XXXX 相对参照的得分差，以及 viz/figures/sentry27_v2_<RULES>/ 下的两张图。
3. 行为差异：rl 和参照在 deaths、no-ammo、on own central、on enemy base 上的区别，以及发现的任何可疑行为。
4. 训练健康度：entropy、ent_coef、approx_kl 的走势；实际吞吐（每小时局数、内存峰值、总耗时、errors 和 stale 计数）。
5. 提交以下文件并推送到 claude/inspiring-lamport-zd97ge 分支：
   - rm_runs/sentry27_v2_<RULES>/latest/policy.pt 和 spec.json
   - rm_runs/sentry27_v2_<RULES>/train_log.csv
   - rm_runs/sentry27_v2_<RULES>_eval/summary.md、summary.json、results.jsonl
   - viz/figures/sentry27_v2_<RULES>/ 下的两张图
   rm_runs/ 和 *.pt 在 .gitignore 里，这几个文件要用 git add -f 添加。
```

## 可选：移植自己队的行为树

如果你们有现役哨兵行为树，可以把它的决策逻辑交给服务器 agent 移植：

1. 在 `rm_rl/arena27/bt.py` 里写一个函数 `f(inp, wps, last_supply=...)`，输入快照，输出各动作头的选项名，并加进 `TREES`。
2. 评估时加上 `--arm bt:<名字> --ref bt:<名字>`。

"胜过我们自己的行为树"比"胜过通用行为树"更有说服力。
