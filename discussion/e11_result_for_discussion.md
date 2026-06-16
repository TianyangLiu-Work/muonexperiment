# result for discussion

## 当前讨论主线

当前 paper 应写成 **head-to-tail interference mechanism paper**：在长尾小批量训练中，head-only update 可能在 tail 样本缺席时扰动这些 tail examples 上的 logits；我们研究 idealized spectral/polar direction 是否能在 matched head gain 下减少这种扰动。

旧的 Muon/Adam condition-geometry 结果只作为 guardrail：它提醒我们不能把 spectral/rank geometry 直接写成优化器全局更优、最终 tail accuracy 更好，或完整 Muon 机制已经被解释。

## 会议讨论点

| 讨论点                                     | 发现                                                                                                                                        | 证据                                                                         | 解释                                                                                                    |
|:----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| G 和 A 的当前定义                             | 主文中严格使用 head gradient 与 downstream tail sensitivity；旧 MF/condition-score 诊断只作背景。                                                          | paper main.tex 的 matched-head-gain / B_T D A_T 定义，以及当前四组 head-to-tail CSV。 | 这样可以避免把旧 A_i proxy 和当前 tail activation product 混在一起。                                                  |
| Synthetic boundary                      | 正例 squared drift ratio=0.3403，反例 squared drift ratio=7.208。                                                                               | results/e11_head_tail_interference/pair_summary.csv                        | nrank(G_H) 与 ssrank(B_T,A_T) 的不等式至少有可 falsify 的符号含义。                                                  |
| One-step long-tail drift                | 20 seeds 下 spectral/Frobenius squared tail-example logit drift ratio=0.5501 [0.5101, 0.5931]。                                             | results/e11_long_tail_one_step/pair_summary.csv                            | matched head gain 下 spectral/polar 对 held-out tail examples 上的 logits 扰动更小。                           |
| Muon-style compatibility                | polar(M_t) squared drift ratio=0.8199 [0.6951, 0.9672]；NS(M_t) squared drift ratio=0.9116 [0.7696, 1.08]。                                 | results/e11_long_tail_muon_bridge/pair_summary.csv                         | momentum polar 是 selected-state compatibility check；finite Newton-Schulz 近似较弱，不能直接推出完整 Muon training。 |
| Practical Muon trajectory compatibility | 120 个 state-step comparisons 下 polar(M_t) squared drift ratio=0.7292 [0.6891, 0.7717]；NS(M_t) squared drift ratio=0.8019 [0.7583, 0.848]。 | results/e11_long_tail_practical_muon_bridge/summary.csv                    | 短 trajectory 上 Muon-style directions 与局部 polar mechanism 兼容，但仍不是 final performance benchmark。         |
| 性能 caveat                               | one-step tail loss diff=0.001399 [0.0002379, 0.002559]。                                                                                   | results/e11_long_tail_one_step/pair_summary.csv                            | 当前结果支持 function drift claim，不支持 tail accuracy / final performance claim。                              |
| 8-step forgetting                       | final squared drift ratio=0.6167 [0.5744, 0.6622]；area ratio=0.7787 [0.7549, 0.8033]。                                                     | results/e11_long_tail_forgetting/summary.csv                               | drift reduction 不只是单步现象，但仍然只是短程诊断。                                                                    |
| Layerwise mechanism                     | layer 1 unit/scaled/observed=1.45/0.4766/0.4767；layer 2=1.476/0.596/0.596。                                                                | results/e11_long_tail_layerwise/summary.csv                                | 机制应写成 matched-head-gain scaling efficiency，而不是“spectral direction 本身更不扰动 tail”。                       |

## 建议会议结论

1. 主文只讲 head-to-tail function drift，不讲 broad optimizer leaderboard。
2. 定理和图都围绕 matched-head-gain protocol 组织。
3. Muon-style compatibility 已从 fixed checkpoint 推进到 short practical trajectory；可以说 selected-state 兼容性证据更多，但仍不能说完整 practical Muon training 或 final tail accuracy。
4. 如果要更强 empirical paper，下一步不是再画旧 geometry 图，而是加 real long-tail benchmark 和 larger-architecture layerwise diagnostic。
