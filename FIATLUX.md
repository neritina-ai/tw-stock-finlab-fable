# 依據 Finlab 免費資料發展 portfolio 投資策略

我有申請免費的 Finlab 帳號，請你運用各種 SOTA 的量化交易技術，開發一套 portfolio 投資策略

## 資源

### tw_stock.db

S:\tw_stock.db 中有台股各個股日 K 線歷史資料

S:\README.md 有注意事項

### Finlab database

取用資料的範例程式

```
import os
import finlab

finlab.login(os.getenv('FINLAB_API_TOKEN'))

from finlab import data

close = data.get('price:收盤價')
```

其它可用資料請參考：

[資料描述](https://ai.finlab.tw/database) 

注意有標示 free 的才能取用，免費用戶流量限制為每日 500 MB，因此應該要先撰寫程式將資料下載至本地再使用

### 其它

這台機器有 GPU，若要安裝 pytorch 可用 nvidia-smi 查詢規格

## 停止條件

這是一個長研究的工作，期待你經過一次又一次的獨立嘗試，終於產生用 Walk Forward validation 驗證出每階段 Sharpe 皆大於 2 的策略。此為停止條件

由於不一定能研究得出來，若已嘗試 20 種方法了，也可以停止

其它時刻皆不需要停下來問我決策，因為你才是專家。請你自行決定

## 每次嘗試

我們從開始到停止以前都是在同一個 session 中，每次新的嘗試開始都是在 /compact 過的上下文

每次嘗試一開始，先想好這次要嘗試的提高投資績效的方法，取一個研究名稱

以此研究名稱建立 git branch，基於前面某個 branch 去擴展不同的變化，形成樹狀結構 (若為全新想法則從 root 分出 branch)

撰寫下載資料的程式來執行 (若沒有使用到新資料則跳過)

實作、回測、超參數優化

為這次的嘗試留下研究紀錄，在研究目錄中以研究名稱建立一個 .md 檔，說明這次嘗試的改進方法是什麼，結果如何，結果中要讓我看得出 portfolio 持股的變化，重點是 Sharpe 離目標還有多遠。然後在研究目錄的 README.md 檔中，將紀錄內容濃縮成一句話，並以超連結連到那個 .md 檔

將這次嘗試及紀錄 commit 進 git，有進步或值得注意的版本，可以加上 tag

施展 self-compact 技能，為下一次嘗試清理出較乾淨的上下文，必要時再回讀一下本檔案以加強印象

若嘗試結果有達到停止條件，用 telegram 通知我，然後停止

### 注意事項

如有儲存資料，例如下載或自行產生後存檔，請在子目錄中的 README.md 中說明各檔案的內容為何，在哪裡用到

當次用到的 temp scripts 須留下來讓 clone 的人能重現，可在當次的研究檔中提到，例如怎樣執行，用來做什麼事。但以前產生將來用不到的 temp scripts 要刪除

