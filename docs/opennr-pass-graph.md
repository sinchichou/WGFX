# OpenNR 1pass 網路 — 精確 Pass 圖與執行語義

> 來源：`effects_src/OpenNR-ReShade/upstream/src/opennr_addon.cpp` 第 2247-2347 行（執行迴圈）、
> `opennr_model.hpp`（層表與 buffer 尺寸）、`opennr_shaders.hpp`（HLSL 核心）。
> 本檔為 task-3 / task-4 的**規格正本**；與本檔衝突者以本檔為準。

## 1. 關鍵結論（先前不確定、現已定案）

| 問題 | 定案 | 依據 |
|---|---|---|
| 哪些層用 GELU？ | **除 `out` 層外，全部 18 個 conv 皆 `act=true`** | addon.cpp:2299, 2281, 2317, 2340 皆傳 `true`；2346 `act=false` |
| bias 是否 f16？ | **否，bias 是 f32（`StructuredBuffer<float4>`）** | shaders.hpp:19；model.hpp:158 `read(".b.bin", pad*4)`、170 以 stride 16 建 buffer |
| w.bin 讀取位元組數 | `9 * cin * pad * 2`（**用 pad 不是 cout**） | model.hpp:157 |
| b.bin 讀取位元組數 | `pad * 4`（f32） | model.hpp:158 |
| pad ≠ cout 的情況 | 僅**輸出層**（`read_conv(last=true)` 時允許 `b2 != pad`） | model.hpp:137-141 |
| w buffer 元素數 | `9 * cin * pad / 4`，stride 8 | model.hpp:164 |
| b buffer 元素數 | `pad / 4`，stride 16 | model.hpp:170 |
| 網路解析度 | `align_extent(n) = max(320, ceil(n/64)*64)` | addon.cpp:50-53 |
| `v = (c - 0.5) * 2 * (1/16)` | 色彩正規化（16 通道輸入用） | shaders.hpp:226 |
| 噪聲時域 | `s = mix32(x*0x9e3779b9 ^ y*0x85ebca6b ^ frame*0xc2b2ae35)` | shaders.hpp:207 |
| Tone/Structure 陷阱 | 傳入的是 `bits(f(g_tone)/100.0f)`，除以 100 在 CPU 做 | addon.cpp:2240 |

## 2. 每層 byte 數驗算（由 model.hpp 公式獨立推導）

`w.bin = 9 * cin * pad * 2`，`b.bin = pad * 4`。以 1pass 為例：

| 層 | cin | cout | pad | level | w.bin 期望 | 實測 | b.bin 期望 | 實測 |
|---|---|---|---|---|---|---|---|---|
| enc0a | 16 | 16 | 16 | 0 | 4608 | 4608 ✓ | 64 | 64 ✓ |
| enc0b | 16 | 16 | 16 | 0 | 4608 | 4608 ✓ | 64 | 64 ✓ |
| enc1a | 16 | 32 | 32 | 1 | 9216 | 9216 ✓ | 128 | 128 ✓ |
| enc1b | 32 | 32 | 32 | 1 | 18432 | 18432 ✓ | 128 | 128 ✓ |
| enc2a | 32 | 64 | 64 | 2 | 36864 | 36864 ✓ | 256 | 256 ✓ |
| enc2b | 64 | 64 | 64 | 2 | 73728 | 73728 ✓ | 256 | 256 ✓ |
| enc3a | 64 | 128 | 128 | 3 | 147456 | 147456 ✓ | 512 | 512 ✓ |
| enc3b | 128 | 128 | 128 | 3 | 294912 | 294912 ✓ | 512 | 512 ✓ |
| mid_a | 128 | 128 | 128 | 3 | 294912 | 294912 ✓ | 512 | 512 ✓ |
| mid_b | 128 | 128 | 128 | 3 | 294912 | 294912 ✓ | 512 | 512 ✓ |
| dec3a | 256 | 128 | 128 | 3 | 589824 | 589824 ✓ | 512 | 512 ✓ |
| dec3b | 128 | 128 | 128 | 3 | 294912 | 294912 ✓ | 512 | 512 ✓ |
| dec2a | 192 | 64 | 64 | 2 | 221184 | 221184 ✓ | 256 | 256 ✓ |
| dec2b | 64 | 64 | 64 | 2 | 73728 | 73728 ✓ | 256 | 256 ✓ |
| dec1a | 96 | 32 | 32 | 1 | 55296 | 55296 ✓ | 128 | 128 ✓ |
| dec1b | 32 | 32 | 32 | 1 | 18432 | 18432 ✓ | 128 | 128 ✓ |
| dec0a | 48 | 16 | 16 | 0 | 13824 | 13824 ✓ | 64 | 64 ✓ |
| dec0b | 16 | 16 | 16 | 0 | 4608 | 4608 ✓ | 64 | 64 ✓ |
| out | 16 | **4** | **16** | 0 | 4608 | 4608 ✓ | 64 | 64 ✓ |

**層表 19 層全部驗算通過** → layout 公式與 model.hpp 一致，task-2 可依此實作。
注意 `out` 層：cin=16, cout=4, pad=16 → 輸出 4 通道但 buffer 寬度 16（補零至 16 的倍數）。

global 分支：`g0 16 32 32 3`、`g1 32 64 64 4`、`g2 64 64 64 5`（level 3/4/5，注意 level 4、5 超出常規 0-3）。
`fc.bin = gH*gi + gH + go*gH + go`（f32）= 64*64 + 64 + 128*64 + 128 = 4096+64+8192+128 = 12480 floats = 49920 bytes ✓ 與實測相符。

## 3. 精確 Pass 序列（1pass，依 addon.cpp:2247-2346）

設網路解析度 `W×H`，`base=16`，`levels=4`，`cin=16`，`cout=4`。

```
P0  stage    : src(W_out×H_out) → chans[16ch @ W×H]         (8×8 threads, 2D dispatch)
--- global 分支（模型有才跑，讀 chans，先用 A/B 不佔 skip）---
G1  pool ×3  : 依序 2x2 平均下採樣 3 次（W→W/2→W/4→W/8），通道固定 16
G2  conv g0  : cin16 cout32 pad32 @ W/8, act=1, zero_pad=1
G3  pool     : → W/16
G4  conv g1  : cin32 cout64 pad64 @ W/16, act=1, zero_pad=1
G5  pool     : → W/32
G6  conv g2  : cin64 cout64 pad64 @ W/32, act=1, zero_pad=1
G7  global   : 1 workgroup：空間均值 → fc1 → gelu → fc2 → += mid_a 的 b  → gbias[128]
--- encoder ---
E1  enc0a @ L0(W×H)   act=1 → B
E2  enc0b @ L0        act=1 → skip[0]        (cur = skip[0])
E3  pool  → W/2
E4  enc1a @ L1(W/2)   act=1 → B
E5  enc1b @ L1        act=1 → skip[1]
E6  pool  → W/4
E7  enc2a @ L2(W/4)   act=1 → B
E8  enc2b @ L2        act=1 → skip[2]
E9  pool  → W/8
E10 enc3a @ L3(W/8)   act=1 → B
E11 enc3b @ L3        act=1 → skip[3]
--- middle（第一個吃 gbias）---
M1  mid_a @ L3        act=1, bias = gbias（有 global 時）
M2  mid_b @ L3        act=1
--- decoder（i = 3→0）---
D1  upcat @ L3 : lo=cur(W/8) skip=skip[3] scale=1 → cat(128,128) = 256ch
D2  dec3a @ L3 act=1
D3  dec3b @ L3 act=1
D4  upcat @ L2 : scale=2 → cat(64,64) = 128ch... （dec2a cin=192 = 128 + 64）
D5  dec2a @ L2 act=1
D6  dec2b @ L2 act=1
D7  upcat @ L1 : scale=2 →（dec1a cin=96 = 64 + 32）
D8  dec1a @ L1 act=1
D9  dec1b @ L1 act=1
D10 upcat @ L0 : scale=2 →（dec0a cin=48 = 32 + 16）
D11 dec0a @ L0 act=1
D12 dec0b @ L0 act=1
--- 輸出 ---
O1  out @ L0 : cin16 cout4 pad16, act=0 → head[4ch @ W×H]（buffer 寬 16）
--- composite（獨立 pass，於輸出解析度）---
C1  composite @ W_out×H_out : 讀 src(raw) + head → correction → finish
```

### upcat 的 cin 分解（關鍵，容易寫錯）
`c0.cin = clo + cs`，其中 `cs = base << i`（skip 通道數）、`clo = c0.cin - cs`（來自下層的上採樣通道數）。
驗算：dec3a i=3 → cs=16<<3=128，clo=256-128=128 ✓；dec2a i=2 → cs=64，clo=192-64=128 ✓；
dec1a i=1 → cs=32，clo=96-32=64 ✓；dec0a i=0 → cs=16，clo=48-16=32 ✓。
→ 上採樣來源通道數 clo 依序為 128, 128, 64, 32。

### upcat 的 scale
`scale = (i+1 < levels) ? 2 : 1`。i=3 時 4<4 為 false → scale=1（L3 同解析度，不放大）。
i=2,1,0 → scale=2。**這解釋了為何 dec3a 前不需放大**。

## 4. dispatch 尺寸公式

- conv: `threads = (W2*H2/4) * (pad/4)`，`groups = ceil(threads/64)`
  （因為 TM=4 像素/執行緒，且每執行緒負責一個 cout4 欄）
- pool: `n = (W2/2)*(H2/2)*(C/4)`，`groups = ceil(n/64)`
- upcat: `n = sw*sh*c0.cin`，`groups = ceil(n/64)`（注意用 cin 不是 cout4）
- global: `groups = 1`（單一 workgroup，128 threads）
- stage / composite: 2D，`(X+7)/8 × (Y+7)/8`

## 5. WGSL 端對應的實作決策（給 task-3）

WGFX 的 `//! DISPATCH <expr>` 支援數值運算式（defines + OUTPUT_WIDTH/HEIGHT + 四則運算）。
因此上述公式可直接寫成：
```
//! DISPATCH ((NW*NH/4)*(PAD/4)+63)/64, 1
```
其中 NW/NH/PAD 由 external defines 提供（task-1 負責注入為 WGSL 常數）。

**中間特徵圖配置**：全部預先配置為具名 storage buffer（由宿主依 task-2 清單配置），
命名建議 `A` / `B`（ping-pong）與 `S0..S3`（skip）、`CH`（輸入 16ch）、`HEAD`（輸出）、`GB`（gbias）。
ping-pong 的切換在 WGFX 中需以不同 pass 讀寫不同 buffer 名稱表達（無執行期別名）。

**重要**：WGFX 目前每個 pass 讀寫的 buffer 名稱是靜態的，所以 ping-pong 必須在
產生效果檔時就展開成明確的 `A`/`B` 交替命名，不能依賴執行期 swap。
