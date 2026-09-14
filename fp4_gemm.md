# BÁO CÁO ĐẶC TẢ CHUYÊN SÂU: TOÁN HỌC, VI KIẾN TRÚC VÀ DATAFLOW CỦA HÀM `fp4_gemm`
*(Phiên bản chuẩn hóa 100% Mapping theo Sơ đồ Vi kiến trúc Draw.io & Mã nguồn Kernel)*

> **Mục đích tài liệu:** Tài liệu này trình bày chi tiết và chuẩn xác tuyệt đối các công thức toán học, định dạng nhị phân, cơ chế phân khối phần cứng (Hardware Tiling), luồng dữ liệu (Dataflow) và ánh xạ mã nguồn của hàm **`fp4_gemm`**. Toàn bộ tên biến, tensor, kích thước (shapes) và ký hiệu trong tài liệu đã được **đồng bộ hóa 100% với sơ đồ Draw.io XML Spec** (`Engine Core`, `xₑ`, `xq`, `scale_x`, `W1`, `W3`, `W2`, `gate`, `up`, `z_fp32`, `zq`, `scale_z`, `contribution`, `FIFO`) và mã nguồn thực thi (`model.py`, `kernel.py`).

---

## MỤC LỤC
1. [Bảng Ánh xạ Toàn diện: Sơ đồ Draw.io vs Hàm `fp4_gemm` vs Mã nguồn PyTorch](#1-bảng-ánh-xạ-toàn-diện-sơ-đồ-drawio-vs-hàm-fp4_gemm-vs-mã-nguồn-pytorch)
2. [Chi tiết Tham số, Kích thước Tensor & Dung lượng Bộ nhớ theo Sơ đồ XML](#2-chi-tiết-tham-số-kích-thước-tensor--dung-lượng-bộ-nhớ-theo-sơ-đồ-xml)
   - [2.1. Nhánh Phase W1 (Gate Projection 4096 → 2048)](#21-nhánh-phase-w1-gate-projection-4096--2048)
   - [2.2. Nhánh Phase W3 (Up Projection 4096 → 2048) & Kỹ thuật "Reuse xq"](#22-nhánh-phase-w3-up-projection-4096--2048--kỹ-thuật-reuse-xq)
   - [2.3. Nhánh Phase W2 (Down Projection 2048 → 4096)](#23-nhánh-phase-w2-down-projection-2048--4096)
3. [Cơ sở Toán học & Thuật toán Nhân Ma trận Lượng tử hóa](#3-cơ-sở-toán-học--thuật-toán-nhân-ma-trận-lượng-tử-hóa)
   - [3.1. Phép nhân ma trận theo chiều rút gọn K (Dot Product)](#31-phép-nhân-ma-trận-theo-chiều-rút-gọn-k-dot-product)
   - [3.2. Công thức khử lượng tử hóa (Dequantization) với Scale Block](#32-công-thức-khử-lượng-tử-hóa-dequantization-với-scale-block)
   - [3.3. Tối ưu đưa Scale ra ngoài tích vô hướng con (Scale Factoring & Fusing)](#33-tối-ưu-đưa-scale-ra-ngoài-tích-vô-hướng-con-scale-factoring--fusing)
   - [3.4. Cấu trúc bit nhị phân IEEE 754: FP4 và UE8M0](#34-cấu-trúc-bit-nhị-phân-ieee-754-fp4-và-ue8m0)
4. [Vi kiến trúc Tiling Phần cứng (Hardware Tiling Breakdown)](#4-vi-kiến-trúc-tiling-phần-cứng-hardware-tiling-breakdown)
   - [4.1. Lưới phân khối 2D Grid [Grid_X, Grid_Y]](#41-lưới-phân-khối-2d-grid-grid_x-grid_y)
   - [4.2. Phân bổ bộ nhớ SRAM (Shared Memory) và Thanh ghi Frag trong PE](#42-phân-bổ-bộ-nhớ-sram-shared-memory-và-thanh-ghi-frag-trong-pe)
   - [4.3. Pipeline vòng lặp K và cơ chế nhịp Scale bất đối xứng (k vs k // 4)](#43-pipeline-vòng-lặp-k-và-cơ-chế-nhịp-scale-bất-đối-xứng-k-vs-k--4)
   - [4.4. Quy trình giải nén FP4 → FP8 và nhân Tensor Core](#44-quy-trình-giải-nén-fp4--fp8-và-nhân-tensor-core)
5. [Ví dụ Số học Cực kỳ Trực quan Xuyên suốt](#5-ví-dụ-số-học-cực-kỳ-trực-quan-xuyên-suốt)
   - [5.1. Ví dụ số học Mini-Vector 4 phần tử](#51-ví-dụ-số-học-mini-vector-4-phần-tử)
   - [5.2. Ví dụ thực tế Token #10 qua Phase W1 và Phase W2](#52-ví-dụ-thực-tế-token-10-qua-phase-w1-và-phase-w2)
6. [Sơ đồ Trực quan Thiết kế cho Draw.io (Mapping Khớp từng Node ID)](#6-sơ-đồ-trực-quan-thiết-kế-cho-drawio-mapping-khớp-từng-node-id)
7. [Mã nguồn TileLang Chuẩn hóa theo Tên biến Draw.io](#7-mã-nguồn-tilelang-chuẩn-hóa-theo-tên-biến-drawio)

---

## 1. BẢNG ÁNH XẠ TOÀN DIỆN: SƠ ĐỒ DRAW.IO VS HÀM `fp4_gemm` VS MÃ NGUỒN PYTORCH

Bảng dưới đây thiết lập mối quan hệ chính xác 100% giữa các thực thể xuất hiện trên sơ đồ **Draw.io XML Spec**, lời gọi hàm **`fp4_gemm`** và mã nguồn **`model.py` / `kernel.py`**:

| Node ID trên Draw.io | Tên khối trên Draw.io XML | Tham số trong hàm `fp4_gemm(x, s, weight, weight.scale)` | Tên biến toán học & Kích thước thực tế | Kiểu dữ liệu (Dtype) | Dung lượng bộ nhớ phần cứng |
|:---:|:---|:---|:---|:---:|:---:|
| **Node 3** | **A2 · QX activation quantizer** | Đầu vào `x`, `s` cho Phase W1 & W3 | $x_q \in \mathbb{R}^{T_e \times 4096}$<br/>$scale\_x \in \mathbb{R}^{T_e \times 32}$ | `float8_e4m3fn`<br/>`float8_e8m0fnu` | $4\text{ KiB} + 32\text{ B}$ / token-expert |
| **Node 2 & 11** | **WTB response** (W1 tiles) | Tham số `weight`, `weight.scale` của $W_1$ | $W_1 \in \mathbb{R}^{2048 \times 4096}$ (logical)<br/>$scale\_w_1 \in \mathbb{R}^{2048 \times 128}$ | `float4_e2m1fn_x2`<br/>`float8_e8m0fnu` | Payload: $4\text{ MiB}$<br/>Scale: $256\text{ KiB}$ |
| **Node 4** | **A3a · PE phase W1** (Gate proj $4096 \to 2048$) | `gate = fp4_gemm(xq, scale_x, W1, scale_w1)` | Đầu ra $gate \in \mathbb{R}^{T_e \times 2048}$ | `bfloat16` | $4\text{ KiB}$ / token-expert |
| **Node 2 & 10** | **WTB response** (W3 tiles) | Tham số `weight`, `weight.scale` của $W_3$ | $W_3 \in \mathbb{R}^{2048 \times 4096}$ (logical)<br/>$scale\_w_3 \in \mathbb{R}^{2048 \times 128}$ | `float4_e2m1fn_x2`<br/>`float8_e8m0fnu` | Payload: $4\text{ MiB}$<br/>Scale: $256\text{ KiB}$ |
| **Node 5** | **A3b · PE phase W3** (Up proj $4096 \to 2048$) | `up = fp4_gemm(xq, scale_x, W3, scale_w3)` | Đầu ra $up \in \mathbb{R}^{T_e \times 2048}$ | `bfloat16` | $4\text{ KiB}$ / token-expert |
| **Node 6** | **A4 · FP32 nonlinear + route** | Tính phi tuyến SwiGLU & Fusing route | $z_{\text{fp32}} = \text{route\_gate} \times \text{SiLU}(\text{gate}) \times \text{up}$<br/>Kích thước: $[T_e, 2048]$ | `float32` | $8\text{ KiB}$ / expert assignment |
| **Node 7** | **A5 · Cast + QZ** | Đầu vào `x`, `s` cho Phase W2 | $z_q \in \mathbb{R}^{T_e \times 2048}$<br/>$scale\_z \in \mathbb{R}^{T_e \times 16}$ | `float8_e4m3fn`<br/>`float8_e8m0fnu` | $2\text{ KiB} + 16\text{ B}$ / expert assignment |
| **Node 2 & 12** | **WTB response** (W2 tiles) | Tham số `weight`, `weight.scale` của $W_2$ | $W_2 \in \mathbb{R}^{4096 \times 2048}$ (logical)<br/>$scale\_w_2 \in \mathbb{R}^{4096 \times 64}$ | `float4_e2m1fn_x2`<br/>`float8_e8m0fnu` | Payload: $4\text{ MiB}$<br/>Scale: $256\text{ KiB}$ |
| **Node 8** | **A6 · PE phase W2** (Down proj $2048 \to 4096$) | `contribution = fp4_gemm(zq, scale_z, W2, scale_w2)` | Đầu ra $contribution \in \mathbb{R}^{T_e \times 4096}$ | `bfloat16` | $8\text{ KiB}$ / token-expert |
| **Node 9 & 27** | **Expert result FIFO** | Gói tin đẩy ra FIFO `{token_id, slot_id, expert_id, contribution}` | $contribution \in \mathbb{R}^{T_e \times 4096}$ | `bfloat16` | Đóng gói theo Packet |

---

## 2. CHI TIẾT THAM SỐ, KÍCH THƯỚC TENSOR & DUNG LƯỢNG BỘ NHỚ THEO SƠ ĐỒ XML

Trong kiến trúc DeepSeek MoE, ký hiệu **$T_e$** đại diện cho **số lượng token được bộ định tuyến Router phân bổ cho Expert hiện tại $e$** trong sequence length $T$.

```
                              TỔNG THỂ DATAFLOW BÊN TRONG ENGINE CORE
                             
  [Node 3: A2 QX]                [Node 4: A3a PE W1]               [Node 6: A4 SwiGLU]
 ┌─────────────────┐            ┌───────────────────┐            ┌──────────────────────┐
 │ x_e [T_e, 4096] │──(quant)──►│ xq [T_e, 4096] FP8│            │ gate = min(gate, +10)│
 │ scale_x [T_e,32]│            │ W1 [2048, 4096]   │──(gate)───►│ up   = clamp(up, ±10)│
 └────────┬────────┘            └───────────────────┘            │ z = route * SiLU * up│
          │                               ▲                      └──────────┬───────────┘
          │ (reuse xq)                    │ (W1 tiles)                      │ z_fp32 [T_e, 2048]
          │                     [Node 2: WTB response]                      ▼
          │                     payload 4 MiB · 256 KiB          [Node 7: A5 Cast + QZ]
          │                               │                      ┌──────────────────────┐
          │                               ▼ (W3 tiles)           │ zq [T_e, 2048] FP8   │
          │                     ┌───────────────────┐            │ scale_z [T_e, 16]    │
          └────────────────────►│ xq [T_e, 4096] FP8│            └──────────┬───────────┘
                                │ W3 [2048, 4096]   │──(up)───────┘          │ (zq + scale_z)
                                └───────────────────┘                        ▼
                                 [Node 5: A3b PE W3]             [Node 8: A6 PE W2]
                                                                 ┌──────────────────────┐
                                 [Node 9: Result FIFO]           │ zq [T_e, 2048] FP8   │
                                ┌─────────────────────┐          │ W2 [4096, 2048]      │
                                │ contribution [4096] │◄─────────│ contribution [4096]  │
                                └─────────────────────┘ (BF16)   └──────────────────────┘
```

---

### 2.1. Nhánh Phase W1 (Gate Projection 4096 → 2048)
- **Lời gọi hàm thực thi**:
  ```python
  gate = fp4_gemm(xq, scale_x, W1, scale_w1, scale_dtype=torch.float8_e8m0fnu)
  ```
- **Kích thước và kiểu dữ liệu từng tham số**:
  - `xq`: Tensor Activation đã lượng tử hóa FP8, shape **$[T_e, 4096]$**, dung lượng $T_e \times 4096\text{ bytes} = \mathbf{4\text{ KiB/token}}$.
  - `scale_x`: Tensor Activation Scale UE8M0, shape **$[T_e, 32]$** ($4096 / 128 = 32$ groups), dung lượng $\mathbf{32\text{ bytes/token}}$.
  - `W1`: Tensor Trọng số nén FP4 từ WTB, shape lưu trữ vật lý $[2048, 2048]$ bytes, shape logic **$[2048, 4096]$**, dung lượng $\mathbf{4\text{ MiB}}$.
  - `scale_w1`: Tensor Weight Scale UE8M0, shape **$[2048, 128]$** ($4096 / 32 = 128$ groups), dung lượng $\mathbf{256\text{ KiB}}$.
  - **Đầu ra `gate`**: Tensor kích thước **$[T_e, 2048]$** kiểu `bfloat16`.

---

### 2.2. Nhánh Phase W3 (Up Projection 4096 → 2048) & Kỹ thuật "Reuse xq"
- **Kỹ thuật "Reuse xq" (Node 13 & 14 trên Draw.io)**: Cả hai mảng PE của $W_1$ và $W_3$ đều sử dụng chung một đầu vào kích hoạt $x_q [T_e, 4096]$ và $scale\_x [T_e, 32]$. Quá trình lượng tử hóa QX tại Node 3 chỉ chạy đúng 1 lần duy nhất, dữ liệu được phát (multicast) đồng thời sang cả hai PE.
- **Lời gọi hàm thực thi**:
  ```python
  up = fp4_gemm(xq, scale_x, W3, scale_w3, scale_dtype=torch.float8_e8m0fnu)
  ```
- **Kích thước và kiểu dữ liệu từng tham số**:
  - `xq`, `scale_x`: Dùng chung với nhánh W1 ($[T_e, 4096]$ FP8 và $[T_e, 32]$ UE8M0).
  - `W3`: Tensor Trọng số nén FP4 từ WTB, shape logic **$[2048, 4096]$**, dung lượng $\mathbf{4\text{ MiB}}$.
  - `scale_w3`: Tensor Weight Scale UE8M0, shape **$[2048, 128]$**, dung lượng $\mathbf{256\text{ KiB}}$.
  - **Đầu ra `up`**: Tensor kích thước **$[T_e, 2048]$** kiểu `bfloat16`.

---

### 2.3. Nhánh Phase W2 (Down Projection 2048 → 4096)
- Sau khi `gate` và `up` được hòa trộn qua hàm SwiGLU tại Node 6:
  $$z_{\text{fp32}} = \text{route\_gate} \times \text{SiLU}(\text{gate}) \times \text{up} \in \mathbb{R}^{T_e \times 2048}$$
  Tensor $z_{\text{fp32}}$ được ép kiểu sang BF16 rồi lượng tử hóa QZ tại Node 7 để tạo thành $z_q$ và $scale\_z$.
- **Lời gọi hàm thực thi**:
  ```python
  contribution = fp4_gemm(zq, scale_z, W2, scale_w2, scale_dtype=torch.float8_e8m0fnu)
  ```
- **Kích thước và kiểu dữ liệu từng tham số**:
  - `zq`: Tensor Activation FP8 trung gian, shape **$[T_e, 2048]$**, dung lượng $T_e \times 2048\text{ bytes} = \mathbf{2\text{ KiB/expert assignment}}$.
  - `scale_z`: Tensor Activation Scale UE8M0, shape **$[T_e, 16]$** ($2048 / 128 = 16$ groups), dung lượng $\mathbf{16\text{ bytes/expert assignment}}$.
  - `W2`: Tensor Trọng số nén FP4 từ WTB, shape logic **$[4096, 2048]$**, dung lượng $\mathbf{4\text{ MiB}}$.
  - `scale_w2`: Tensor Weight Scale UE8M0, shape **$[4096, 64]$** ($2048 / 32 = 64$ groups), dung lượng $\mathbf{256\text{ KiB}}$.
  - **Đầu ra `contribution`**: Tensor kích thước **$[T_e, 4096]$** kiểu `bfloat16`, chuẩn bị nạp vào FIFO (Node 9) để tích lũy tại Node 22 (A7 Accumulator).

---

## 3. CƠ SỞ TOÁN HỌC & THUẬT TOÁN NHÂN MA TRẬN LƯỢNG TỬ HÓA

### 3.1. Phép nhân ma trận theo chiều rút gọn K (Dot Product)
Hàm `fp4_gemm` thực hiện phép nhân ma trận tổng quát dạng:
$$\mathbf{Output = Activation \times Weight^T}$$

#### Đối với Phase W1 (Gate) & Phase W3 (Up):
$$\mathbf{gate[t, n]} = \sum_{k=0}^{4095} x_{\text{real}}[t, k] \times W_{1\text{, real}}[n, k]$$
- $t \in [0, T_e - 1]$: Chỉ số token trong batch của Expert.
- $n \in [0, 2047]$: Chỉ số neuron ẩn trung gian đầu ra.
- $k \in [0, 4095]$: Chiều rút gọn reduction $K = 4096$.

#### Đối với Phase W2 (Down):
$$\mathbf{contribution[t, m]} = \sum_{k=0}^{2047} z_{\text{real}}[t, k] \times W_{2\text{, real}}[m, k]$$
- $t \in [0, T_e - 1]$: Chỉ số token.
- $m \in [0, 4095]$: Chiều đặc trưng token đầu ra.
- $k \in [0, 2047]$: Chiều rút gọn reduction $K = 2048$.

```
           W1^T [4096, 2048] (hoặc W3^T, W2^T)
                  Neuron thứ n
                  ┌───────────┐
                  │  W[n, 0]  │
                  │  W[n, 1]  │
                  │    ...    │  (K phần tử dọc chiều reduction)
                  │ W[n, K-1] │
                  └─────┬─────┘
                        │
xq [T_e, 4096]          │
Token thứ t             ▼
┌─────────────────────────┐    ┌──────────────────────────────────────────────┐
│ xq[t, 0] ... xq[t, K-1] │ ──►│ gate[t, n] = dot(xq[t, :], W1[n, :]) (BF16) │
└─────────────────────────┘    └──────────────────────────────────────────────┘
```

---

### 3.2. Công thức khử lượng tử hóa (Dequantization) với Scale Block
Các giá trị số thực lý thuyết được tái tạo từ giá trị lượng tử hóa thông qua các hệ số Scale khối:
1. **Activation $x_q$** (Nhóm $K_{\text{group}} = 128$ phần tử):
   $$x_{\text{real}}[t, k] = x_q[t, k] \times scale\_x\left[t, \left\lfloor \frac{k}{128} \right\rfloor\right]$$
2. **Weight $W_1$** (Nhóm $K_{\text{group}} = 32$ phần tử):
   $$W_{1\text{, real}}[n, k] = W_{1\text{, fp4}}[n, k] \times scale\_w_1\left[n, \left\lfloor \frac{k}{32} \right\rfloor\right]$$

Thế vào công thức tích vô hướng:
$$gate[t, n] = \sum_{k=0}^{4095} \left( x_q[t, k] \times scale\_x\left[t, \left\lfloor \frac{k}{128} \right\rfloor\right] \right) \times \left( W_{1\text{, fp4}}[n, k] \times scale\_w_1\left[n, \left\lfloor \frac{k}{32} \right\rfloor\right] \right)$$

---

### 3.3. Tối ưu đưa Scale ra ngoài tích vô hướng con (Scale Factoring & Fusing)
Để loại bỏ việc phải nhân scale tại từng phần tử $k$, chiều $K$ được băm nhỏ thành các sub-block kích thước đúng bằng **$block\_K = 32$**.
- Với $K=4096$, có $4096 / 32 = \mathbf{128\text{ sub-blocks}}$ (đánh chỉ số $p = 0 \dots 127$).
- Trong phạm vi 32 phần tử của sub-block thứ $p$ ($k = 32p \dots 32p + 31$):
  - $scale\_w_1[n, p]$ là hằng số đối với neuron $n$.
  - $scale\_x[t, \lfloor p / 4 \rfloor]$ là hằng số đối với token $t$ (do $128 / 32 = 4$).

Do đó, hai hệ số scale được rút ra ngoài tổng tích vô hướng của 32 phần tử:

$$\mathbf{gate[t, n]} = \sum_{p=0}^{127} \Bigg( \underbrace{\left[ \sum_{j=0}^{31} x_q[t, 32p + j] \times W_{1\text{, fp8\_decoded}}[n, 32p + j] \right]}_{\text{Tính toán trên FP8 Tensor Core (gate\_local)}} \times \underbrace{scale\_x\left[t, \left\lfloor \frac{p}{4} \right\rfloor\right]}_{\text{Act Scale (group 128)}} \times \underbrace{scale\_w_1[n, p]}_{\text{Weight Scale (group 32)}} \Bigg)$$

> **Quy tắc vàng phần cứng:**
> 1. Tensor Core thực hiện nhân ma trận FP8 $\times$ FP8 thuần túy với tốc độ tối đa vào thanh ghi tích lũy cục bộ.
> 2. Cứ sau mỗi block 32 phần tử, mạch phần cứng nhân hệ số gộp $(scale\_x \times scale\_w)$ đúng 1 lần duy nhất rồi tích lũy vào thanh ghi FP32 Accumulator.

---

### 3.4. Cấu trúc bit nhị phân IEEE 754: FP4 và UE8M0

#### A. Trọng số FP4 (`float4_e2m1fn`)
- Gồm: **1 bit Dấu ($S$)**, **2 bit Số mũ ($E$)**, **1 bit Định trị ($M$)**, Bias $= 1$.
- Bảng ánh xạ 16 trạng thái nhị phân sang số thực:

| Mã Hex | Chuỗi Bit ($S\,E_1E_0\,M$) | Giá trị số thực | Ghi chú phần cứng |
|:---:|:---:|:---:|:---|
| `0x0` | `0 00 0` | $+0.0$ | Số không dương |
| `0x1` | `0 00 1` | $+0.5$ | Số subnormal ($2^0 \times 0.5$) |
| `0x2` | `0 01 0` | $+1.0$ | Số normal ($2^0 \times 1.0$) |
| `0x3` | `0 01 1` | $+1.5$ | Số normal ($2^0 \times 1.5$) |
| `0x4` | `0 10 0` | $+2.0$ | Số normal ($2^1 \times 1.0$) |
| `0x5` | `0 10 1` | $+3.0$ | Số normal ($2^1 \times 1.5$) |
| `0x6` | `0 11 0` | $+4.0$ | Số normal ($2^2 \times 1.0$) |
| `0x7` | `0 11 1` | $+6.0$ | Giá trị dương cực đại của FP4 |
| `0x8` – `0xF` | `1 ...` | $-0.0 \dots -6.0$ | Đối xứng hoàn toàn mang dấu âm |

#### B. Scale UE8M0 (`float8_e8m0fnu`)
- Gồm: **8 bit Số mũ không dấu (Unsigned Exponent)**, Bias $= 127$. Không có bit Mantissa, không có bit Dấu.
- Luôn là lũy thừa nguyên cơ số 2:
  $$\text{Scale} = 2^{\text{byte\_value} - 127}$$
- *Ví dụ:*
  - Byte nhị phân `01111000` (Thập phân 120): $2^{120 - 127} = 2^{-7} = \frac{1}{128} \approx 0.0078125$.
  - Byte nhị phân `01111111` (Thập phân 127): $2^{127 - 127} = 2^{0} = 1.0$.

---

## 4. VI KIẾN TRÚC TILING PHẦN CỨNG (HARDWARE TILING BREAKDOWN)

### 4.1. Lưới phân khối 2D Grid [Grid_X, Grid_Y]
Phần cứng chia ma trận kết quả thành các tile kích thước cố định:
- **`block_Te = 32`**: Mỗi Thread Block xử lý một bó gồm 32 tokens ($T_e$).
- **`block_N = 128`**: Mỗi Thread Block tính toán cho 128 neuron đầu ra.
- **`block_K = 32`**: Bước nhảy dọc theo chiều reduction $K$.

#### Cấu hình Grid cho từng Phase:
- **Phase W1 & Phase W3** ($N = 2048, K = 4096$):
  $$\text{Grid\_X} = \frac{2048}{128} = \mathbf{16\text{ tiles}}, \quad \text{Grid\_Y} = \left\lceil \frac{T_e}{32} \right\rceil$$
- **Phase W2** ($N = 4096, K = 2048$):
  $$\text{Grid\_X} = \frac{4096}{128} = \mathbf{32\text{ tiles}}, \quad \text{Grid\_Y} = \left\lceil \frac{T_e}{32} \right\rceil$$

```
                                KHÔNG GIAN GRID 2D CỦA PE ARRAY
                    bx = 0             bx = 1                 bx = Grid_X - 1
               ┌──────────────────┬──────────────────┬───┬──────────────────┐
  by = 0       │ Tile [32, 128]   │ Tile [32, 128]   │...│ Tile [32, 128]   │
  Tokens 0..31 │ Neuron 0..127    │ Neuron 128..255  │...│ Neuron N-128..N-1│
               ├──────────────────┼──────────────────┼───┼──────────────────┤
  by = 1       │ Tile [32, 128]   │ Tile [32, 128]   │...│ Tile [32, 128]   │
  Tokens 32..63│ Neuron 0..127    │ Neuron 128..255  │...│ Neuron N-128..N-1│
               └──────────────────┴──────────────────┴───┴──────────────────┘
```

---

### 4.2. Phân bổ bộ nhớ SRAM (Shared Memory) và Thanh ghi Frag trong PE
Bên trong mỗi Core xử lý một Tile $[32, 128]$, các bộ đệm được ánh xạ chính xác theo bảng sau:

| Tên bộ đệm phần cứng | Kiểu dữ liệu | Shape thực tế | Dung lượng | Chức năng chi tiết |
|:---|:---:|:---:|:---:|:---|
| **`xq_shared`** (hoặc `zq_shared`) | FP8 | $[32, 32]$ | $1024\text{ Bytes}$ ($1\text{ KiB}$) | Chứa sub-tile Activation của 32 tokens tại bước $k$ |
| **`W_fp4_shared`** | FP4 packed | $[128, 16]$ Bytes | $2048\text{ Bytes}$ ($2\text{ KiB}$) | Nhận 128 kênh $\times$ 32 phần tử nén từ bộ đệm WTB |
| **`W_shared`** | FP8 | $[128, 32]$ | $4096\text{ Bytes}$ ($4\text{ KiB}$) | Trọng số sau khi giải nén on-the-fly sang FP8 |
| **`scale_x_frag`** (hoặc `scale_z_frag`)| FP32 | $[32]$ | $128\text{ Bytes}$ | Thanh ghi lưu scale Activation cho 32 tokens |
| **`scale_w_frag`** | FP32 | $[128]$ | $512\text{ Bytes}$ | Thanh ghi lưu scale Weight cho 128 neurons |
| **`local_frag`** | FP32 | $[32, 128]$ | $16\text{ KiB}$ | Thanh ghi lưu kết quả tích vô hướng FP8 $\times$ FP8 của bước $k$ |
| **`accum_frag`** | FP32 | $[32, 128]$ | $16\text{ KiB}$ | Thanh ghi tích lũy Fused Scale cộng dồn qua mọi chu kỳ $k$ |

---

### 4.3. Pipeline vòng lặp K và cơ chế nhịp Scale bất đối xứng (k vs k // 4)
Số vòng lặp pipeline:
- **Phase W1 & W3**: $K_{\text{iters}} = 4096 / 32 = \mathbf{128\text{ chu kỳ}}$ ($k = 0 \dots 127$).
- **Phase W2**: $K_{\text{iters}} = 2048 / 32 = \mathbf{64\text{ chu kỳ}}$ ($k = 0 \dots 63$).

#### Cơ chế lệch pha Scale (Asymmetric Scale Stride):
Trong mã nguồn `kernel.py:500-505`:
```python
# Weight scale thay đổi liên tục sau MỖI bước k (bước nhảy = 32 phần tử)
scale_w_frag[i] = scales_w[bx * block_N + i, k]

# Activation scale chỉ thay đổi sau MỖI 4 bước k (bước nhảy = 128 phần tử: k // 4)
scale_x_frag[i] = scales_x[by * block_Te + i, k // 4]
```

```
Chu kỳ Pipeline k:     0    1    2    3  │   4    5    6    7  │  ...  │ 124  125  126  127
                     ───  ───  ───  ───  ┼  ───  ───  ───  ───  ┼  ───  ┼ ───  ───  ───  ───
Chỉ số Scale Weight:   0    1    2    3  │   4    5    6    7  │  ...  │ 124  125  126  127  (Nhảy liên tục)
Chỉ số Scale Act:      0    0    0    0  │   1    1    1    1  │  ...  │  31   31   31   31   (Cố định 4 bước)
```

---

### 4.4. Quy trình giải nén FP4 → FP8 và nhân Tensor Core
Trong mỗi chu kỳ pipeline $k$, các thao tác phần cứng diễn ra theo đúng chuỗi:

```
                  WTB Memory Response
                          │
                          │ 1. Load Tile W [128, 16 bytes]
                          ▼
            ┌───────────────────────────┐
            │ W_fp4_shared [128, 16 B]  │
            └─────────────┬─────────────┘
                          │ 2. Unpack từng Byte -> 2 số FP4
                          ▼
            ┌───────────────────────────┐
            │ LUT Bit-Unpack & Cast     │
            │ FP4 -> FP32 -> FP8 (e4m3) │
            └─────────────┬─────────────┘
                          │ 3. Lưu vào Shared Memory
                          ▼
┌───────────────────────────┐       ┌───────────────────────────┐
│   W_shared [128, 32] FP8  │       │  xq_shared [32, 32] FP8   │
└─────────────┬─────────────┘       └─────────────┬─────────────┘
              │                                   │
              └─────────────────┬─────────────────┘
                                │ 4. Tensor Core GEMM: xq @ W^T
                                ▼
              ┌───────────────────────────────────┐
              │ local_frag [32, 128] (FP32)       │
              └─────────────────┬─────────────────┘
                                │
    scale_x_frag [32] ──────────┼────────── scale_w_frag [128]
                                │ 5. Fused Multiply-Accumulate
                                ▼
              ┌───────────────────────────────────┐
              │ accum_frag [32, 128] +=           │
              │   local_frag * scale_x * scale_w  │
              └───────────────────────────────────┘
```

---

## 5. VÍ DỤ SỐ HỌC CỰC KỲ TRỰC QUAN XUYÊN SUỐT

### 5.1. Ví dụ số học Mini-Vector 4 phần tử
Xét phép tính giữa 1 Token ($T_e = 1$) và 1 Neuron ($N = 1$) với $K = 4$ phần tử:
- **Activation thực**: $x_{\text{real}} = [+1.5,\; -3.0,\; +0.5,\; +2.0]$
- **Trọng số thực**: $W_{\text{real}} = [+0.5,\; +2.0,\; -1.0,\; +6.0]$

#### Bước 1: Lượng tử hóa QX (Node 3 trên Draw.io)
- $\text{amax} = 3.0 \implies scale\_x = 2^{-5} = 0.03125$.
- Vector FP8:
  $$x_q = \frac{[+1.5,\; -3.0,\; +0.5,\; +2.0]}{0.03125} = [+48,\; -96,\; +16,\; +64]$$

#### Bước 2: Lượng tử hóa WTB (Node 2 trên Draw.io)
- $scale\_w = 2^0 = 1.0$.
- Các giá trị FP4: $W_{\text{fp4}} = [+0.5,\; +2.0,\; -1.0,\; +6.0]$.
  - Mã Hex nhị phân: `[0x1, 0x4, 0xA, 0x7]`.
  - Đóng gói thành 2 bytes: `Byte_0 = 0x41` (chứa $0.5$ và $2.0$); `Byte_1 = 0x7A` (chứa $-1.0$ và $6.0$).

#### Bước 3: Tính toán trong `fp4_gemm` (Node 4: PE Phase W1)
1. **Unpack & Cast sang FP8**:
   - Byte `0x41` tách thành $0.5$ và $2.0$.
   - Byte `0x7A` tách thành $-1.0$ và $6.0$.
   - Mảng FP8: $W_{\text{fp8}} = [+0.5,\; +2.0,\; -1.0,\; +6.0]$.
2. **Nhân Tensor Core ($x_q \times W_{\text{fp8}}$)**:
   $$\text{local\_frag} = (+48 \times 0.5) + (-96 \times 2.0) + (+16 \times -1.0) + (+64 \times 6.0)$$
   $$\text{local\_frag} = 24.0 - 192.0 - 16.0 + 384.0 = \mathbf{200.0}$$
3. **Nhân Fused Scale vào thanh ghi Accumulator**:
   $$\text{accum\_frag} = \text{local\_frag} \times scale\_x \times scale\_w = 200.0 \times 0.03125 \times 1.0 = \mathbf{6.25}$$

#### Bước 4: Kiểm chứng với số thực ban đầu
$$\text{Dot Product gốc} = (1.5 \times 0.5) + (-3.0 \times 2.0) + (0.5 \times -1.0) + (2.0 \times 6.0) = \mathbf{6.25}$$
$\implies$ **Khớp chính xác tuyệt đối 100%!**

---

### 5.2. Ví dụ thực tế Token #10 qua Phase W1 và Phase W2

Theo đúng kịch bản của `routed_core.md`, xét **Token #10** được Router gán cho **Expert #137**:

#### A. Tại Phase W1 (Node 4: Gate Projection 4096 → 2048)
- **Input**:
  - $x_q$: Vector $[1, 4096]$ FP8 ($4\text{ KiB}$).
  - $scale\_x$: $32$ giá trị UE8M0 ($32\text{ bytes}$).
  - $W_1$: Ma trận $[2048, 4096]$ FP4 ($4\text{ MiB}$).
  - $scale\_w_1$: Ma trận $[2048, 128]$ UE8M0 ($256\text{ KiB}$).
- **Xử lý Tiling**:
  - Lưới Grid: $\text{Grid\_X} = 2048 / 128 = 16\text{ tiles}$; $\text{Grid\_Y} = 1\text{ tile}$.
  - Mỗi tile chạy qua $128$ chu kỳ pipeline $k$ ($4096 / 32 = 128$).
- **Output**: Vector `gate` kích thước $[1, 2048]$ kiểu BF16 (giả sử phần tử thứ 0 ra $+4.2$).

#### B. Tại Phase W3 (Node 5: Up Projection 4096 → 2048)
- Chạy song song với Phase W1, tái sử dụng cùng $x_q$ và $scale\_x$.
- **Output**: Vector `up` kích thước $[1, 2048]$ kiểu BF16 (giả sử phần tử thứ 0 ra $-1.8$).

#### C. Tại Phase SwiGLU & QZ (Node 6 & Node 7)
- Kẹp biên: $\text{gate} = \min(4.2, 10.0) = 4.2$; $\text{up} = \text{clamp}(-1.8, -10.0, 10.0) = -1.8$.
- Fusing route: $z_{\text{fp32}} = 0.18 \times \text{SiLU}(4.2) \times (-1.8) \approx -\mathbf{1.3407}$.
- QZ Quantizer sinh ra $z_q [1, 2048]$ FP8 ($2\text{ KiB}$) và $scale\_z [1, 16]$ UE8M0 ($16\text{ bytes}$).

#### D. Tại Phase W2 (Node 8: Down Projection 2048 → 4096)
- **Input**: $z_q [1, 2048]$, $scale\_z [1, 16]$, $W_2 [4096, 2048]$ FP4, $scale\_w_2 [4096, 64]$.
- **Xử lý Tiling**:
  - Lưới Grid: $\text{Grid\_X} = 4096 / 128 = 32\text{ tiles}$; $\text{Grid\_Y} = 1\text{ tile}$.
  - Mỗi tile chạy qua $64$ chu kỳ pipeline $k$ ($2048 / 32 = 64$).
- **Output**: Vector `contribution` kích thước $[1, 4096]$ kiểu BF16.
- Đẩy vào FIFO (Node 9) với gói tin `{token_id: 10, slot_id: 2, expert_id: 137, contribution: [...]}`.

---

## 6. SƠ ĐỒ TRỰC QUAN THIẾT KẾ CHO DRAW.IO (MAPPING KHỚP TỪNG NODE ID)

Dưới đây là sơ đồ chi tiết biểu diễn cấu trúc nội bộ của khối PE Phase W1/W3/W2 để bạn ghép trực tiếp vào sơ đồ Draw.io XML:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ [Node 4 / Node 5 / Node 8] : PE Phase Engine (fp4_gemm_kernel)                         │
│                                                                                        │
│ CÁC CỔNG VÀO (INPUTS)                                      CỔNG RA (OUTPUT)            │
│ ┌────────────────────────────────────┐                     ┌─────────────────────────┐ │
│ │ xq / zq FP8 [T_e, K]               │                     │ gate / up / contribution│ │
│ │ (W1/W3: [T_e, 4096] | W2: [T_e,2048])│                    │ BF16 [T_e, N]           │ │
│ └────────────────────────────────────┘                     │ (W1/W3: [T_e, 2048]     │ │
│ ┌────────────────────────────────────┐                     │  W2:    [T_e, 4096])    │ │
│ │ scale_x / scale_z UE8M0 [T_e, K/128│                     └─────────────────────────┘ │
│ └────────────────────────────────────┘                                  ▲              │
│ ┌────────────────────────────────────┐                                  │              │
│ │ W1 / W3 / W2 FP4 payload (4 MiB)   │                                  │              │
│ └────────────────────────────────────┘                                  │              │
│ ┌────────────────────────────────────┐                                  │              │
│ │ scale_w UE8M0 (256 KiB)            │                                  │              │
│ └────────────────────────────────────┘                                  │              │
│                                                                         │              │
│ VI KIẾN TRÚC THỰC THI NỘI BỘ (INSIDE PE CORE)                           │              │
│ ┌───────────────────────────────────────────────────────────────────────┴────────────┐ │
│ │ VÒNG LẶP PIPELINE THEO K (k = 0 ... K/32 - 1)                                      │ │
│ │                                                                                    │ │
│ │ 1. Load SRAM:                                                                      │ │
│ │    xq_shared [32, 32] FP8  ◄── Load từ xq                                          │ │
│ │    W_fp4_shared [128, 16B] ◄── Load từ WTB                                         │ │
│ │                                                                                    │ │
│ │ 2. Unpack Unit:                                                                    │ │
│ │    W_shared [128, 32] FP8  ◄── Unpack bitwise & Cast FP4 -> FP8                    │ │
│ │                                                                                    │ │
│ │ 3. Scale Fragments:                                                                │ │
│ │    scale_w_frag [128]      ◄── scales_w[:, k]          (nhảy mỗi bước k)           │ │
│ │    scale_x_frag [32]       ◄── scales_x[:, k // 4]     (nhảy mỗi 4 bước k)         │ │
│ │                                                                                    │ │
│ │ 4. Tensor Core GEMM:                                                               │ │
│ │    local_frag [32, 128] = xq_shared [32, 32] @ W_shared [128, 32]^T (FP32)         │ │
│ │                                                                                    │ │
│ │ 5. Fused Scale Accumulator:                                                        │ │
│ │    accum_frag [32, 128] += local_frag * scale_x_frag * scale_w_frag                │ │
│ └────────────────────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. MÃ NGUỒN TILELANG CHUẨN HÓA THEO TÊN BIẾN DRAW.IO

Đoạn mã dưới đây là mã nguồn chuẩn xác trong `kernel.py:441-516` đã được cập nhật đồng bộ các tên biến theo sơ đồ Draw.io:

```python
@tilelang.jit(pass_configs=pass_configs)
def fp4_gemm_kernel(N, K, out_dtype=BF16, accum_dtype=FP32, scale_dtype=FP32):
    """
    FP8 Activation x FP4 Weight GEMM Kernel
    
    Phù hợp cho cả 3 Phase:
    - Phase W1 (Gate): K = 4096, N = 2048, out = gate BF16 [T_e, 2048]
    - Phase W3 (Up):   K = 4096, N = 2048, out = up BF16 [T_e, 2048]
    - Phase W2 (Down): K = 2048, N = 4096, out = contribution BF16 [T_e, 4096]
    """
    T_e = T.symbolic("T_e")
    act_group_size = 128
    weight_group_size = 32
    block_Te = 32     # Mỗi block xử lý 32 tokens
    block_N = 128     # Mỗi block xử lý 128 neuron đầu ra
    block_K = 32      # Bước nhảy reduction dọc theo K
    n_sub = act_group_size // block_K  # = 4 sub-blocks trên 1 nhóm scale activation

    @T.prim_func
    def fp4_gemm_kernel_(
        xq: T.Tensor[(T_e, K), FP8],                                        # Activation FP8 [T_e, 4096]
        W: T.Tensor[(N, K), FP4],                                          # Trọng số FP4 [N, 4096]
        out: T.Tensor[(T_e, N), out_dtype],                                # Output BF16 [T_e, N]
        scales_x: T.Tensor[(T_e, T.ceildiv(K, act_group_size)), scale_dtype],    # Act Scale [T_e, 32]
        scales_w: T.Tensor[(N, T.ceildiv(K, weight_group_size)), scale_dtype],  # Weight Scale [N, 128]
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(T_e, block_Te), threads=128) as (bx, by):
            # Cấp phát SRAM trên Chip (Shared Memory)
            xq_shared = T.alloc_shared((block_Te, block_K), FP8)           # [32, 32] FP8 (1 KiB)
            W_fp4_shared = T.alloc_shared((block_N, block_K), FP4)         # [128, 16] Bytes (2 KiB)
            W_shared = T.alloc_shared((block_N, block_K), FP8)             # [128, 32] FP8 (4 KiB)
            
            # Cấp phát Thanh ghi Nội bộ (Registers Fragment)
            local_frag = T.alloc_fragment((block_Te, block_N), accum_dtype)
            accum_frag = T.alloc_fragment((block_Te, block_N), accum_dtype)
            scale_x_frag = T.alloc_fragment((block_Te,), FP32)
            scale_w_frag = T.alloc_fragment((block_N,), FP32)

            T.clear(local_frag)
            T.clear(accum_frag)

            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=2):
                # 1. Load Tile từ Bộ nhớ ngoài vào Shared Memory
                T.copy(xq[by * block_Te, k * block_K], xq_shared)
                T.copy(W[bx * block_N, k * block_K], W_fp4_shared)

                # 2. Giải nén FP4 -> FP8 thông qua FP32
                for i, j in T.Parallel(block_N, block_K):
                    W_shared[i, j] = T.Cast(FP8, T.Cast(FP32, W_fp4_shared[i, j]))

                # 3. Load Scale Trọng số (bước nhảy k)
                for i in T.Parallel(block_N):
                    scale_w_frag[i] = T.Cast(FP32, scales_w[bx * block_N + i, k])

                # 4. Load Scale Activation (bước nhảy k // 4)
                for i in T.Parallel(block_Te):
                    scale_x_frag[i] = T.Cast(FP32, scales_x[by * block_Te + i, k // n_sub])

                # 5. Phép nhân ma trận Tensor Core
                T.gemm(xq_shared, W_shared, local_frag, transpose_B=True)

                # 6. Hòa trộn Scale Fused vào FP32 Accumulator
                for i, j in T.Parallel(block_Te, block_N):
                    accum_frag[i, j] += local_frag[i, j] * scale_x_frag[i] * scale_w_frag[j]
                T.clear(local_frag)

            # 7. Ghi kết quả cuối cùng ra bộ nhớ BF16
            T.copy(accum_frag, out[by * block_Te, bx * block_N])
```

---

## 8. TỔNG KẾT

Tài liệu này đã hoàn thiện việc chuyển đổi toàn bộ hệ thống ký hiệu sang chuẩn vi kiến trúc:
1. **Khớp 100% tên biến**: Sử dụng trực tiếp $x_e$, $x_q$, $scale\_x$, $W_1$, $W_3$, $W_2$, $scale\_w$, `gate`, `up`, $z_{\text{fp32}}$, $z_q$, $scale\_z$, `contribution`, `FIFO` tương ứng chính xác với từng Node trên sơ đồ Draw.io.
2. **Khớp 100% kích thước**: Sử dụng kích thước thực tế $[T_e, 4096]$, $[2048, 4096]$, $[T_e, 2048]$, $[4096, 2048]$, dung lượng $4\text{ MiB}, 256\text{ KiB}, 4\text{ KiB} + 32\text{ B}$.
3. **Sẵn sàng để vẽ**: Bạn có thể dùng trực tiếp cấu trúc bảng, các sơ đồ ASCII và mã XML trong tài liệu này để cập nhật hoặc mở rộng sơ đồ Draw.io của mình một cách hoàn hảo và chuyên nghiệp nhất.
