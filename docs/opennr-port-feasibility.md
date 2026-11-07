# OpenNR → WGFX 移植可行性評估（研究結論）

> 依據：`effects_src/OpenNR-ReShade/OpenNR-ReShade-研究报告.md`（v0.2.1 逆向報告）
> 結論日期：2026-11-08　|　工作分支：`opennr-port`

## 判定：可行，且 WGFX 是合適載體

| 面向 | 評估 |
|---|---|
| 模型 | 1.29M 參數 / 2.58MB / 19 層 3×3 conv f16 U-Net，51.2 GMAC @ 704p，35 dispatch/幀 |
| 算子 | 全部標準（Conv/AveragePool/Resize/Concat/GELU/Add），無自訂數值語義 |
| 授權 | MIT（含權重）；教師模型不在倉庫，無第三方 IP 風險 |
| WGFX 適配 | 多 pass compute 管線正是 WGFX 的核心場景（Anime4K_Restore_Soft_UL 已 16 pass） |
| 效能預估 | dGPU f16 ~4–8 ms/幀；核顯 ~30–60 ms（報告 4.3 節，未實測） |

## 與報告 V1/V2/V3 路線的對應

報告建議 V1=ORT-Web、V2=int8、V3=手寫 WGSL。**WGFX 對應 V3 路線**：
效果 DSL 逐 pass 對應網路段（stage/global/enc×4/mid/dec×4/out/composite），
權重以 **f16 紋理外部資源** 餵入（`externalResources.textures` 已支援），
宿主（YouTube VSR 擴充）載入轉換後的權重紋理。

## 需要的前置工作

1. **取得上游**：`github.com/MakeDecisionWorth/OpenNR-ReShade`（本地僅有報告與分析產物，無權重/HLSL）。
2. **WGFX 框架增強：幀計數器 uniform** —— 噪聲通道種子 = `mix32(x·… ⊕ y·… ⊕ frame·…)`
   逐幀變化（報告 3.2 節）；SceneInfo 需增加 `frame: u32`（40B → 48B buffer），
   `updateSceneBuffer` 增加可選參數，向後相容。
3. **權重轉換器**（Node 腳本）：`model.txt` + `*.w.bin`/`*.b.bin`/`fc.bin`
   → f16 紋理（`[tap][cin][cout/4]` 佈局原樣保留，WGSL 端 `textureLoad` 解碼），
   輸出 JSON 清單供宿主載入。
4. **效果檔**：`effects/OpenNR/OpenNR_1pass.wgsl`（~20 pass）；
   global 分支 7 dispatch 可融合為 1（報告 2.3 節）。
5. **composite**：0.25 係數 / Strength / 三風格 grade（14 參數，純函式直接移植）；
   2pass 變體 12 通道輸出按風格選組。

## 驗收

- 單幀數值對照：WGFX 產物 vs 報告附錄的 C++ 參考語義逐通道 ≤1e-2（f16 容差）。
- 噪聲時域：相鄰兩幀噪型逐幀變化（frame 進種子）、停格輸入輸出穩定。
- 觀感：704p 影片 Local Tone 100% 下對照原插件截圖（arch.png / VSR_on.png 場景）。

## 風險（報告 4.4 節摘錄 + 新增）

- 噪聲種子若被向量化改變 → 時域觀感劣化（保留 frame 進哈希）。
- ±1/16 輸入動態範圍對 f16/量化敏感 → 色帶驗收。
- 704 行為訓練分佈上限 → 按原設計先降採樣再推理。
- WGFX 尚無 16 通道中間紋理的一等支援 → 以多張 rgba16float 紋理打包
  （Anime4K 移植既有模式）。
