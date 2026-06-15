# tw-stock-finlab — 台股 Portfolio 量化策略研究

以 Finlab 免費資料 + 本地台股日 K 資料庫 (`S:\tw_stock.db`)，研究 Walk-Forward 驗證下
每階段 Sharpe > 2 的 portfolio 投資策略。任務說明見 [FIATLUX.md](FIATLUX.md)。

## 開發紀錄

- 研究紀錄索引：[research/README.md](research/README.md)
- AI model：本專案由 **Claude Fable 5** (Anthropic) 開發。

## 結構

```
src/twq/        共用程式庫（資料載入、回測引擎、Walk-Forward 框架）
scripts/        資料下載與工具腳本
research/       每次嘗試的研究紀錄 (.md)
data/           本地資料快取（gitignored；parquet）
```

## 環境

```
python -m venv .venv && .venv\Scripts\activate
pip install -e .
set FINLAB_API_TOKEN=<your token>
python scripts/download_finlab.py
```

開發時實際使用 `uv` 管理（`uv sync`）。

## 資料注意事項

- `S:\tw_stock.db`：原始（未還原權息）日 K；`ID IN ('TWSE','OTC')` 為 ETL 標記列須排除。
- 還原股價：以 Finlab `etl:adj_close` 推得每日還原因子，套用到本地 OHLC。
- 台股 2015-06-01 起漲跌幅 7% → 10%。
- 交易成本：買 0.1425% 手續費；賣 0.1425% 手續費 + 0.3% 交易稅。
- Finlab 免費帳號每日流量 500 MB；資料一律先下載至 `data/finlab/` parquet 再離線使用。
