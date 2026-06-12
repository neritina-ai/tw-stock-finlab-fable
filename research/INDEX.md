# 研究紀錄索引

目標：Walk-Forward validation 每階段 OOS Sharpe > 2（淨成本後）。

| # | 研究 | 一句話結論 |
|---|------|-----------|
| 1 | [px-composite-v1](px-composite-v1.md) | 價量複合因子月頻基線：WFV 1/7 階段達標（OOS Sharpe 0.6-2.1），defensive 因子+regime 避險方向正確但 alpha 與成本皆需改善。 |
| 2 | [band-voltarget](band-voltarget.md) | 週頻 slow_keep banding 把周轉 12x→3-9x（成功、之後沿用），但 WFV 0/6：事後最佳也只 ~1.8，價量 defensive 家族 alpha 天花板已到，須換 ML 排序引擎；breadth gate 訓練期過擬合。 |
| 3 | [lgbm-rank-v1](lgbm-rank-v1.md) | 滾動 LightGBM 排序（31 段 OOS IC 全正、mean 0.103）：WFV 1/6（stage5 2.14 ✓），天花板抬到 2.2 但市場特徵造成組合鞭甩（周轉 10-19x）；下一步拆開排序與曝險。 |
| 4 | [lgbm-stockonly](lgbm-stockonly.md) | 排序/曝險拆離：鞭甩消滅（周轉 4-8x、單週最多換 2 檔）但 WFV 0/6 — 市場特徵的條件 alpha 是真的（天花板 4/6 段下降）；taiex_half 固定選擇穩健（距天花板平均 -0.14）；弱段瓶頸=alpha 量，下一步基本面拼接或雙 horizon。 |
| 5 | [fund-splice](fund-splice.md) | 發現免費層是「只給歷史到 2018-12」：月營收 2005 起/融資 2009/估值 2010/法人 2012 全在磁碟 — +13 基本面籌碼特徵拼接 nomkt + 固定 taiex_half：WFV 1/6（stage4 2.06 ✓）、5/6 段進步、平均 1.14→1.56、stage1 0.36→1.70 打滿天花板；唯天花板仍只 1/6 段 >2，下一步 h63 雙 horizon 或排名混合。 |
| 6 | [h63-ensemble](h63-ensemble.md) | h63 慢排序器（IC 0.128/0.135 > h21）+ 多面板排名平均：all4 mix WFV 2/6（stage2 2.54 ✓、stage4 2.10 ✓）、平均 1.56→1.75、stage3 0.89→1.30、stage2 MaxDD -3.7% — 分歧段混合突破單模天花板，但 stage1 fund 獨強被稀釋 1.70→1.30、stage6 選參缺口 0.63；下一步 IC 自適應權重 + ε 低周轉 tie-break。 |
| 7 | [ic-adaptive](ic-adaptive.md) | 負結果×2（冠軍仍 all4 1.75）：PIT trailing-IC 面板權重把 stage1 天花板 1.30→1.62（機制有效、金融股回流可視化）但選參掉 top50 陷阱只實得 1.12、分歧段天花板被傾斜侵蝕（2.54→2.45）；ε=0.05 低周轉 tie-break 有害（all4 平均 1.75→1.67、stage2 -0.45）— 瓶頸正式轉移到選參層：ic 引擎天花板均 1.93、選參完美可 3/6，下一步 top-k 參數 ensemble + 近期窗選參。 |
| 8 | [param-ensemble](param-ensemble.md) | 負結果（選參層三連攻收官）：top-3 參數 ensemble 僅在選參誤差大處有效（ic stage4 2.02 過線、stage1 +0.15、stage5 新高 1.87）但 sleeve 重疊 ~90% 稀釋稅更大（all4 1.75→1.67）；trailing-5y 選參在 all4 六段選擇完全不變（no-op）、b8/s8 天花板組在任何 train 指標前三皆不可見 — train→OOS 排序邊際是系統性偏誤，誠實選參已榨乾（實得 1.75 vs 天花板 1.89-1.93），等權+b4/s4/top30 即此族誠實上限；attempt 9 改抬天花板：產業中性化排序救 stage 1/3。 |
