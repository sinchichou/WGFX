# WGFX (WebGPU Effect Framework)

![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![Status: Beta](https://img.shields.io/badge/Status-Beta-yellow)
![Platform: Web/Node](https://img.shields.io/badge/Platform-Web%20%7C%20Node.js-green)

**WGFX** 是一個基於 WebGPU 的高效能圖形特效框架。它借鑒了 **Magpie** 的設計思維，旨在為 Web 環境提供一個強大且易用的特效開發與執行平台。WGFX 支援解析自定義的 `.wgsl` 格式特效檔案，並能自動管理 GPU 資源、Pipeline 構建以及多通道渲染。

其核心價值在於協助開發者將原本 HLSL 编写的特效（尤其是 Magpie 或 MPDN 風格）輕鬆遷移至現代 Web 瀏覽器，實現如即時升頻 (Upscaling)、降噪 (Denoising) 等複雜的影像後處理功能。

---

## 🌟 核心特性

- 🚀 **高效能實作**：直接操作 WebGPU API，充分發揮現代 GPU 的平行運算能力。
- 📝 **增強版 WGSL 格式**：透過特殊的元數據註解 (如 `//! PASS`, `//! TEXTURE`)，讓著色器代碼具備自我描述能力，簡化資源宣告與渲染通道設定。
- ⚙️ **自動化資源管理**：自動處理 `GPUTexture`、`GPUBuffer`、`GPUSampler` 的生命週期與繫結組 (Bind Group) 生成。
- 🔄 **Runtime 編譯與優化**：內建基於 Peggy 的解析器，能在執行期動態解析特效檔並根據裝置環境最佳化生成的 WGSL 代碼。
- 🛠️ **跨平台架構**：提供純瀏覽器支持 (ESM/UMD) 並兼容 Node.js 環境（需配合 `@webgpu/node`）。

---

## 🛠️ 技術架構

WGFX 遵循「單一職責原則」，將複雜的 WebGPU 流程拆解為多個核心模組。詳細的設計說明請參閱 [技術架構概覽](docs/technical/OVERVIEW.md)。

### 專案目錄結構

```text
WGFX/
├── src/
│   ├── WGFX.ts             # 主要面向使用者的 API 入口
│   ├── runtime/            # 核心運行時模組
│   │   ├── WGFXRuntime.ts      # 協調整個運行時生命週期的引擎
│   │   ├── ShaderParser.pegjs  # 特效檔案語法定義
│   │   ├── ResourceManager.ts  # GPU 資源 (紋理、緩衝區) 管理
│   │   ├── PipelineManager.ts  # 渲染管線與通道調度中心
│   │   ├── WGSLCodeGenerator.ts# WGSL 代碼生成與預處理器
│   │   └── UniformBinder.ts    # 動態 Uniform 參數繫結工具
│   ├── types/              # 型別定義
│   └── utils/              # 通用工具 (如 Logger)
├── examples/               # 各種 .wgsl 特效範例 (如 Anime4K)
├── debugger/               # 基於 Vite 的開發者測試介面
├── dist/                   # 編譯產物
└── docs/                   # 自動生成的 API 文件
```

---

## 📦 安裝與建置

### 作為相依套件安裝

```bash
npm install wgfx
```

### 從源碼開發

1. **複製專案**：

   ```bash
   git clone https://github.com/sinchichou/WGFX.git
   cd WGFX
   ```

2. **安裝依賴**：

   ```bash
   npm install
   ```

3. **打包編譯**：

   ```bash
   npm run build
   ```

4. **運行偵錯工具**：
   ```bash
   npm run debug
   ```

---

## 🚀 快速上手

### 核心 API 使用範例

```typescript
import { WGFX } from "wgfx";

// 1. 初始化 WebGPU 裝置
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

// 2. 建立 WGFX 實例
const effectCode = await fetch("effects/Anime4K_Upscale.wgsl").then((r) =>
  r.text()
);
const wgfx = await WGFX.create({
  device,
  effectCode,
  width: 1920,
  height: 1080,
});

// 3. 處理影像來源 (支持 Video, ImageBitmap, Canvas 等)
const videoElement = document.querySelector("video");
const outputTexture = await wgfx.process(videoElement);

// 4. 接獲輸出視圖並顯示
const outputView = wgfx.getOutputView();
// 在你的渲染循環中使用 outputView...

// 5. 即時更新參數
wgfx.updateUniforms({ Strength: 1.5 });
```

---

## 📄 .wgsl 特效格式規範

WGFX 的特效檔包含特殊的指示符 (Directives) 與標準 WGSL 代碼：

| 指示符                 | 描述                              | 範例                      |
| :--------------------- | :-------------------------------- | :------------------------ |
| `//! PASS <index>`     | 定義一個渲染通道                  | `//! PASS 1`              |
| `//! TEXTURE <name>`   | 聲明一個中間紋理資源              | `//! TEXTURE TempTex`     |
| `//! PARAMETER <name>` | 定義一個可動態調整的 Uniform 參數 | `//! PARAMETER Sharpness` |
| `//! COMMON`           | 所有通道共享的通用代碼塊          | `//! COMMON`              |

_(詳細規格請參考 [API 文件](docs/modules.md))_

---

## 🙏 致謝 (Credits)

本專案的開發深受以下優秀開源專案與社群的啟發，特此致謝：

1.  **[Magpie](https://github.com/Blinue/Magpie)**: 本專案的核心設計靈感、特效格式以及部分架構理念皆源自 Magpie。感謝作者 **Blinue** 在圖像處理開發領域的卓越貢獻，讓高品質缩放算法能在 Windows 桌面環境發光發熱。
2.  **[Anime4K](https://github.com/bloc97/Anime4K)**: 範例中的高品質即時動畫升頻演算法源自 **bloc97** 的 Anime4K 專案。
3.  **[Peggy](https://peggyjs.org/)**: 強大的解析器生成器，為 WGFX 提供了穩定且靈活的指令解析能力。
4.  **WebGPU Community**: 感謝 W3C GPU 網路工作小組提供的現代圖形規範。

---

## 📜 授權協議

本專案採用 **GNU Affero General Public License v3.0 (AGPL-3.0)** 進行授權。
這意味著如果您在網路服務中使用此程式碼（即使不分發套件），您也必須根據 AGPL 要求開源相關的衍生作品。

詳情請參閱 [LICENSE](LICENSE) 檔案。
