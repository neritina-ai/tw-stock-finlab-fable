# 研究紀錄索引

目標：Walk-Forward validation 每階段 OOS Sharpe > 2（淨成本後）。

| # | 研究 | 一句話結論 |
|---|------|-----------|
| 1 | [px-composite-v1](px-composite-v1.md) | 價量複合因子月頻基線：WFV 1/7 階段達標（OOS Sharpe 0.6-2.1），defensive 因子+regime 避險方向正確但 alpha 與成本皆需改善。 |
| 2 | [band-voltarget](band-voltarget.md) | 週頻 slow_keep banding 把周轉 12x→3-9x（成功、之後沿用），但 WFV 0/6：事後最佳也只 ~1.8，價量 defensive 家族 alpha 天花板已到，須換 ML 排序引擎；breadth gate 訓練期過擬合。 |
| 3 | [lgbm-rank-v1](lgbm-rank-v1.md) | 滾動 LightGBM 排序（31 段 OOS IC 全正、mean 0.103）：WFV 1/6（stage5 2.14 ✓），天花板抬到 2.2 但市場特徵造成組合鞭甩（周轉 10-19x）；下一步拆開排序與曝險。 |
