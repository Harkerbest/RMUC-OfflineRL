# 给服务器上 agent 的提示词

整段复制给服务器上的 agent（例如 Claude Code）即可。

```text
你在一台 Linux 服务器上（4×RTX 5880 Ada、144 核、128 GB 内存）训练 RMUC 2027 哨兵的决策层：
在 Arena-1 仿真里用 PPO 自博弈训练一个策略，让它在配对评估中胜过行为树。
代码、设计和判据都在仓库里，动手前先完整读一遍 docs/sentry27/README.md。

## 准备
1. 克隆仓库并切到分支：
   git clone https://github.com/Harkerbest/RMUC-OfflineRL.git && cd RMUC-OfflineRL
   git checkout claude/inspiring-lamport-zd97ge
   pip install -r requirements.txt
   torch 按服务器驱动装对应的 CUDA 版本。
2. 在仓库外克隆 Arena-1，固定到已验证的提交，再编译 Rust 引擎；没有 Rust 先用 rustup 安装：
   git clone https://github.com/RM-Arena/Arena-1.git ../Arena-1
   cd ../Arena-1 && git checkout bbaa885 && cargo build --release --manifest-path simulation/Cargo.toml
   Arena-1 是非商业学习许可：只作为外部依赖使用，不要复制进本仓库，也不要修改它的代码和规则。
3. 回到仓库，设置环境变量：
   export ARENA1_ROOT=$(realpath ../Arena-1)
4. 自检：python -m rm_rl.arena27.selftest
   必须输出 ALL PASS；有失败就停下，把输出发给我。

## 运行
用 tmux 或 nohup 运行，日志写到文件里。每一步都可以中断后用同一条命令续跑。
1. 基线评估，约 1 小时：
   STAGE=baseline bash scripts/sentry27.sh
2. 训练，默认 300 次更新，约 1.5 天：
   STAGE=train bash scripts/sentry27.sh
3. 最终评估，同时评估 latest 和每 50 次更新的检查点：
   STAGE=eval bash scripts/sentry27.sh
4. 出图：
   python -m viz.sentry27_figures --train rm_runs/sentry27 --eval rm_runs/sentry27_eval --out viz/figures

## 资源
- 训练 worker 数默认为 CPU 数减 8。开跑后前 10 分钟用 free -g 观察：每个 worker 约 0.5 GB，
  内存超过 110 GB 就停掉，用 TRAIN_WORKERS=112 重启（会续跑）。
- 学习器自动用第一张 GPU；其余 GPU 不需要。瓶颈是 CPU。
- 停止训练：只给主进程（python -m rm_rl.train.train_sentry27）发 SIGTERM，它会一并关闭 worker。
  不要用 pkill -f 匹配宽泛的命令行，可能误杀别的进程，包括你自己的 shell。
  停止后检查没有残留的 spawn_main 或 rmuc-sim 进程。

## 训练中每隔几小时查看 rm_runs/sentry27/train_log.csv
- 模仿阶段（前 6 次更新）结束时，imit_acc 应 ≥ 0.9；达不到就加大 --imitate-iters 从头重训。
- PPO 阶段：
  - approx_kl 多数 < 0.03，clipfrac < 0.3；
  - explained_var 应逐步升到 0.5 以上；
  - entropy 平缓变化，不应突然掉到接近 0；
  - kl_prior 是离模仿策略的距离。默认锚定（0.1 → 0.01）是为了防止 PPO 早期把贪心策略推坏；
    如果 100 次更新后 kl_prior 仍接近 0、得分也不动，说明锚太紧，可以用 --prior-kl 0.03 在新目录重训，并在汇报里说明。
- errors 应保持为 0。偶发几次可以；持续增长就停下，看日志里的 "rollout error"。
- score_script / score_bt 是学习者对固定对手的得分（采样策略，含探索），应缓慢上升；score_mirror 应在 0.5 左右。
- 关注 seat_no_ammo、seat_own_central、seat_deaths。行为出现异常时（例如得分上升但 seat_own_central 掉到接近 0，
  或长时间完全不开火），去 results 里查那几局，判断是不是在利用仿真漏洞，并记录下来。
- 想在训练中途看学习曲线，可以少开 24 个训练 worker，用这 24 个核评估检查点：
  python -m rm_rl.eval.eval_sentry27 --out rm_runs/sentry27_eval --workers 24 --seeds 1 \
      --arm bt:balanced --arm rl_0050=rm_runs/sentry27/iter_0050
  也可以等训练结束后统一评估。

## 不要做
- 训练中途不要改观测、动作、奖励或对手组成。这会破坏续跑和可比性。确实要改，就用新的 --out 目录从头训练，并在汇报里写明改了什么、为什么。
- 不要修改 Arena-1，也不要为了让数字好看去改评估网格、参照对象或留出的队伍。
- 不要提交 rm_runs/ 下的 versions/、league/、state.pt。

## 结束后汇报给我
1. rm_runs/sentry27_eval/summary.md 全文，加一句话结论：
   rl 相对 bt:balanced 的配对得分差及 95% 置信区间，下界是否 > 0；rl 相对 script 是多少。
2. 学习曲线：各检查点 rl_XXXX 的配对得分差，以及 viz/figures/sentry27_train.png 和 sentry27_eval.png。
3. 行为差异：rl 和 bt:balanced 在 deaths、no-ammo、on own central 上的区别，以及发现的任何可疑行为。
4. 实际吞吐：每小时局数、内存峰值、总耗时、errors 和 stale 的计数。
5. 提交以下文件并推送到 claude/inspiring-lamport-zd97ge 分支：
   - rm_runs/sentry27/latest/policy.pt 和 spec.json
   - rm_runs/sentry27/train_log.csv
   - rm_runs/sentry27_eval/summary.md、summary.json、results.jsonl
   - viz/figures/sentry27_train.png、sentry27_eval.png
   rm_runs/ 和 *.pt 在 .gitignore 里，这几个文件要用 git add -f 添加。
```

## 可选：移植自己队的行为树

如果你们有现役哨兵行为树，可以把它的决策逻辑交给服务器 agent 移植：

1. 在 `rm_rl/arena27/bt.py` 里写一个函数 `f(inp, wps, last_supply=...)`，输入快照，输出各动作头的选项名，并加进 `TREES`。
2. 评估时加上 `--arm bt:<名字> --ref bt:<名字>`。

"胜过我们自己的行为树"比"胜过通用行为树"更有说服力。
