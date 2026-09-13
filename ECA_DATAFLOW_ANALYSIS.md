# BÁO CÁO PHÂN TÍCH CHUYÊN SÂU: KIẾN TRÚC VÀ DATAFLOW ECA - TOP (DEEPSEEK MoE)

> **Mục đích tài liệu:** Phân tích chi tiết luồng dữ liệu (dataflow), nguyên lý hoạt động, ý nghĩa từng dòng code của cụm **ECA - TOP (Expert Compute Array)** và các khối ngoại vi liên quan trong mô hình DeepSeek (tại `model.py`, `kernel.py`, `config.json`), đồng thời **đối chiếu và so sánh chính xác 100% với đặc tả sơ đồ Draw.io XML**.

---

## MỤC LỤC
1. [Tổng quan kiến trúc ECA - TOP](#1-tổng-quan-kiến-trúc-eca---top)
2. [Sơ đồ Dataflow toàn cục (End-to-End Diagram)](#2-sơ-đồ-dataflow-toàn-cục-end-to-end-diagram)
3. [Bảng theo dõi vòng đời Tensor (Tensor Lifecycle & Shapes)](#3-bảng-theo-dõi-vòng-đời-tensor-tensor-lifecycle--shapes)
4. [Phân tích chi tiết các khối bên trong ECA - TOP](#4-phân-tích-chi-tiết-các-khối-bên-trong-eca---top)
   - [A · Routed Expert Core (4 Engines & 256 Experts)](#a--routed-expert-core-4-engines--256-experts)
   - [B · Shared Expert (FP8 Mode)](#b--shared-expert-fp8-mode)
   - [C · Combine + Completion](#c--combine--completion)
5. [Phân tích chi tiết các khối Ngoại vi (Ngoài ECA)](#5-phân-tích-chi-tiết-các-khối-ngoại-vi-ngoài-eca)
   - [Ngoài ECA · Post-block mHC (Sinkhorn Pre-mix & RMSNorm)](#ngoài-eca--post-block-mhc-sinkhorn-pre-mix--rmsnorm)
   - [Ngoài ECA · MRU / Router (Gate)](#ngoài-eca--mru--router-gate)
   - [Ngoài ECA · WTB / DDR4 (Storage Core)](#ngoài-eca--wtb--ddr4-storage-core)
   - [Ngoài ECA · Residual Mix (MoE)](#ngoài-eca--residual-mix-moe)
6. [Bảng đối chiếu & so sánh Spec (Draw.io) vs Thực tế Source Code](#6-bảng-đối-chiếu--so-sánh-spec-drawio-vs-thực-tế-source-code)
   - [Các điểm khớp hoàn toàn (Exact Matches)](#các-điểm-khớp-hoàn-toàn-exact-matches)
   - [Các điểm khác biệt & chi tiết cần lưu ý đặc biệt (Discrepancies & Nuances)](#các-điểm-khác-biệt--chi-tiết-cần-lưu-ý-đặc-biệt-discrepancies--nuances)
7. [Hướng dẫn đọc code và thực thi từng bước (Step-by-Step Code Walkthrough)](#7-hướng-dẫn-đọc-code-và-thực-thi-từng-bước-step-by-step-code-walkthrough)

---

## 1. Tổng quan kiến trúc ECA - TOP

Trong thiết kế phần cứng vi kiến trúc (Hardware Accelerator/ASIC) và mô hình hóa PyTorch của DeepSeek:
- **ECA** viết tắt của **Expert Compute Array** (mảng tính toán chuyên dụng cho các chuyên gia MoE).
- **ECA - TOP** là khối bao bọc (wrapper block) chịu trách nhiệm nhận đặc trưng token $h$, phân phối tính toán qua **Routed Experts** (được chia cho 4 Engine/Ranks) và **Shared Expert**, sau đó tích lũy kết quả thành tensor $y$.
- **Mục tiêu kiến trúc**:
  1. Đạt được dung lượng tham số khổng lồ (256 routed experts + 1 shared expert) nhưng chỉ tiêu tốn FLOPs của $6 + 1 = 7$ experts cho mỗi token.
  2. Tối ưu hóa băng thông bộ nhớ và FLOPs thông qua tính toán hỗn hợp: **FP4** cho routed experts, **FP8** cho shared expert, tích lũy bằng **FP32 Accumulator**, và trả về **BF16**.

---

## 2. Sơ đồ Dataflow toàn cục (End-to-End Diagram)

Dưới đây là sơ đồ chi tiết biểu diễn luồng dữ liệu từ khối ngoài ECA đi vào ECA - TOP và quay trở lại nhánh Hyper-Connections:

```mermaid
flowchart TD
    subgraph OUT_PRE["Ngoài ECA · Chuẩn bị đầu vào"]
        X_bypass["Residual bypass X'<br/>[T, 4, 4096] BF16<br/>(32 KiB/token)"]
        Post_mHC["Ngoài ECA · Post-block mHC<br/>Sinkhorn collapse (pre) + RMSNorm"]
        h_token["h [T, 4096] BF16 reference<br/>(8 KiB/token · 2 MiB khi T=256)"]
        
        X_bypass --> Post_mHC
        Post_mHC --> h_token
    end

    subgraph OUT_ROUTER["Ngoài ECA · Định tuyến & Bộ nhớ"]
        MRU["Ngoài ECA · MRU / Router (Gate)<br/>topk=6, score_func='sqrtsoftplus'<br/>route_scale=1.5"]
        WTB["Ngoài ECA · WTB / DDR4<br/>(Storage Core ngoài ECA)<br/>W1, W3, W2 payload + scales"]
        
        h_token -->|h [T, 4096]| MRU
        MRU -->|indices [T, 6]<br/>weights [T, 6] FP32| ECA_ROUTED
        WTB -->|routed FP4 tiles| ECA_ROUTED
        WTB -->|shared FP8 weights (24 MiB)| ECA_SHARED
    end

    subgraph ECA_TOP["ECA - TOP (Khung trung tâm)"]
        subgraph ECA_ROUTED["A · Routed Expert Core (256 experts · 6 active/token)"]
            E0["Engine 0<br/>IDs 0–63"]
            E1["Engine 1<br/>IDs 64–127"]
            E2["Engine 2<br/>IDs 128–191"]
            E3["Engine 3<br/>IDs 192–255"]
            
            SwiGLU_path["Đường tính: W1 → W3 → SwiGLU × route_weight → W2<br/>Tagged FP32 accumulation across 6 slots"]
            
            E0 -.-> SwiGLU_path
            E1 -.-> SwiGLU_path
            E2 -.-> SwiGLU_path
            E3 -.-> SwiGLU_path
        end
        
        subgraph ECA_SHARED["B · Shared Expert"]
            Shared_core["Chạy cho MỌI token · Không route_weight<br/>Official FP8-weight mode<br/>W1/W3/W2 payload = 24 MiB"]
        end

        subgraph ECA_COMBINE["C · Combine + Completion"]
            Combine_Node["y_full_fp32 = y_routed_fp32 + y_shared_fp32<br/>Final cast → y_bf16 [T, 4096]"]
        end
        
        h_token -->|h [T, 4096]| ECA_ROUTED
        h_token -->|h [T, 4096]| ECA_SHARED
        SwiGLU_path -->|y_routed_fp32 [T, 4096]<br/>(16 KiB/token)| Combine_Node
        Shared_core -->|y_shared_fp32 [T, 4096]| Combine_Node
    end

    subgraph OUT_POST["Ngoài ECA · Hoàn thiện Residual"]
        y_out["y_bf16 [T, 4096]<br/>(8 KiB/token · 2 MiB khi T=256)"]
        Residual_Mix["Ngoài ECA · Residual Mix (MoE)<br/>X_next = B · X' + C · y<br/>(hc_post function)"]
        X_next["X_next [T, 4, 4096] BF16<br/>Đầu ra chuyển tiếp sang Block tiếp theo"]

        Combine_Node --> y_out
        y_out --> Residual_Mix
        X_bypass -->|Bypass X'| Residual_Mix
        Residual_Mix --> X_next
    end

    classDef ecaBox fill:#ECFDF5,stroke:#059669,stroke-width:2px;
    classDef sharedBox fill:#FFF7ED,stroke:#EA580C,stroke-width:2px;
    classDef combineBox fill:#EFF6FF,stroke:#2563EB,stroke-width:2px;
    classDef outerBox fill:#F5F3FF,stroke:#7C3AED,stroke-width:2px;
    classDef memBox fill:#FFFBEB,stroke:#D97706,stroke-width:2px;

    class ECA_ROUTED ecaBox;
    class ECA_SHARED sharedBox;
    class ECA_COMBINE combineBox;
    class Post_mHC,MRU,Residual_Mix outerBox;
    class WTB memBox;
```

---

## 3. Bảng theo dõi vòng đời Tensor (Tensor Lifecycle & Shapes)

Giả sử batch size $B=1$, độ dài chuỗi $S=T=256$ tokens:

| Bước (Step) | Tên biến trong code | Kiểu dữ liệu (Dtype) | Shape vật lý | Dung lượng bộ nhớ (Memory footprint) | Ý nghĩa / Ghi chú |
|:---|:---|:---|:---|:---|:---|
| **1** | `residual` ($X'$) | BF16 (2 bytes) | `[1, 256, 4, 4096]` | $256 \times 4 \times 4096 \times 2 = 8\text{ MiB}$ (32 KiB/token) | Tensor bypass 4 nhánh Hyper-Connections |
| **2** | $y$ từ `hc_pre` | BF16 (2 bytes) | `[1, 256, 4096]` | $256 \times 4096 \times 2 = 2\text{ MiB}$ (8 KiB/token) | Gom 4 nhánh về 1 nhánh qua Sinkhorn weights |
| **3** | $h$ vào MoE | BF16 (2 bytes) | `[256, 4096]` | $2\text{ MiB}$ (8 KiB/token) | Sau chuẩn hóa `ffn_norm` (RMSNorm) |
| **4** | `indices` (Router) | INT64 / INT32 | `[256, 6]` | $\approx 6\text{ KiB}$ | Chỉ số Top-6 experts được kích hoạt cho mỗi token |
| **5** | `weights` (Router) | FP32 (4 bytes) | `[256, 6]` | $\approx 6\text{ KiB}$ | Trọng số định tuyến sau khi chuẩn hóa và nhân scale 1.5 |
| **6** | `y` (tích lũy MoE) | **FP32** (4 bytes) | `[256, 4096]` | $256 \times 4096 \times 4 = 4\text{ MiB}$ (**16 KiB/token**) | Tagged FP32 Accumulation của các routed experts |
| **7** | `y_shared` | FP32 | `[256, 4096]` | $4\text{ MiB}$ (16 KiB/token) | Kết quả tính toán từ Shared Expert cho toàn bộ 256 tokens |
| **8** | $y_{\text{bf16}}$ (kết thúc MoE)| BF16 (2 bytes) | `[256, 4096]` | $2\text{ MiB}$ (**8 KiB/token**) | Cast ngược lại từ FP32 accumulator về BF16 |
| **9** | $X_{\text{next}}$ (`hc_post`) | BF16 (2 bytes) | `[1, 256, 4, 4096]` | $8\text{ MiB}$ (32 KiB/token) | Kết hợp $B \cdot X' + C \cdot y$ mở rộng lại 4 nhánh |

---

## 4. Phân tích chi tiết các khối bên trong ECA - TOP

Cụm ECA nằm trong `model.py:609-645` (Class `MoE`) và `model.py:587-607` (Class `Expert`).

### A · Routed Expert Core (4 Engines & 256 Experts)
- **Vị trí code**: `model.py:616-625` và `model.py:633-642`
- **Thông số cấu hình**:
  - `n_routed_experts = 256`
  - `n_activated_experts = 6` (Top-6)
  - `world_size = 4` (tương ứng 4 Engines phần cứng)
  - Mỗi engine quản lý: $256 / 4 = 64$ experts:
    - **Engine 0**: IDs `0 – 63`
    - **Engine 1**: IDs `64 – 127`
    - **Engine 2**: IDs `128 – 191`
    - **Engine 3**: IDs `192 – 255`
- **Định dạng trọng số**: `torch.float4_e2m1fn_x2` (FP4 packed, 2 giá trị 4-bit trong 1 byte uint8) đi kèm scale `torch.float8_e8m0fnu` (mỗi nhóm 32 phần tử có 1 hệ số scale 8-bit).

#### Cơ chế điều phối: Expert-Major Scheduling
Trong phần cứng, thay vì chạy từng token qua 6 expert (Token-major), ECA thực hiện **Expert-major**: Nhóm tất cả các token được định tuyến về cùng một expert $i$ để nạp trọng số $W_1, W_3, W_2$ một lần và thực thi GEMM theo batch:
```python
# model.py:634-640
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i) # Lấy danh sách token chọn expert i
    y[idx] += expert(x[idx], weights[idx, top, None]) # Tích lũy FP32
```

#### Đường tính toán: $W_1 \to W_3 \to \text{SwiGLU} \times \text{route\_weight} \to W_2$
Chi tiết luồng tính toán bên trong `Expert.forward` (`model.py:596-606`):
1. **Gate Projection**: $\text{gate} = W_1(x) \in \mathbb{R}^{B_{\text{sub}} \times 2048}$
2. **Up Projection**: $\text{up} = W_3(x) \in \mathbb{R}^{B_{\text{sub}} \times 2048}$
3. **Clamping & Kích hoạt phi tuyến (SwiGLU)**:
   $$\text{SwiGLU}(x) = \text{SiLU}(\text{gate}) \odot \text{up}$$
4. **Nhân trọng số định tuyến trước $W_2$**:
   $$x_{\text{inter}} = \text{route\_weight} \odot \text{SwiGLU}(x)$$
   > **Ý nghĩa kiến trúc cực kỳ quan trọng**:
   > Vì phép chiếu $W_2$ là tuyến tính ($W_2(\alpha \cdot v) = \alpha \cdot W_2(v)$), việc nhân trọng số định tuyến `weights` ngay tại không gian trung gian $2048$ thay vì đợi ra không gian $4096$ giúp:
   > - Giảm một nửa số phép tính nhân scale vector ($2048$ phần tử thay vì $4096$ phần tử).
   > - Tiết kiệm băng thông ghi nhớ thanh ghi tích lũy.
5. **Down Projection**: Trả về $W_2(x_{\text{inter}}) \in \mathbb{R}^{B_{\text{sub}} \times 4096}$.
6. **Tagged FP32 Accumulation**: Cộng dồn trực tiếp vào bộ đệm tích lũy `y[idx]` (kiểu FP32) theo đúng vị trí token (`idx`).

---

### B · Shared Expert (FP8 Mode)
- **Vị trí code**: `model.py:626-627` và `model.py:643`
- **Đặc điểm**:
  - Chạy cho **tất cả mọi token** trong chuỗi $x$ (`T` tokens).
  - Không qua router, không bị nhân bởi `route_weight`.
  - Luôn thường trực trong bộ nhớ SRAM/HBM cục bộ (Official FP8-weight mode).
- **Tính toán Payload bộ nhớ của Shared Expert**:
  - $W_1$: Kích thước $[2048, 4096] \to 2048 \times 4096 = 8{,}388{,}608$ tham số.
  - $W_3$: Kích thước $[2048, 4096] \to 2048 \times 4096 = 8{,}388{,}608$ tham số.
  - $W_2$: Kích thước $[4096, 2048] \to 4096 \times 2048 = 8{,}388{,}608$ tham số.
  - Tổng số tham số $= 3 \times 8{,}388{,}608 = 25{,}165{,}824$ elements.
  - Mỗi tham số lưu ở dạng FP8 (1 byte) $\to 25{,}165{,}824\text{ bytes} = \mathbf{24\text{ MiB}}$!
  *(Con số $24\text{ MiB}$ này khớp chính xác tuyệt đối với ghi chú trên sơ đồ Draw.io).*

---

### C · Combine + Completion
- **Vị trí code**: `model.py:641-644`
- **Cơ chế**:
  1. Nếu chạy song song đa GPU (`world_size > 1`), gọi `dist.all_reduce(y)` để gom kết quả tích lũy từ cả 4 Engine.
  2. Cộng thêm kết quả từ Shared Expert:
     $$y_{\text{full\_fp32}} = y_{\text{routed\_fp32}} + y_{\text{shared\_fp32}}$$
  3. **Completion & Type Cast**: Ép kiểu từ FP32 accumulator ngược lại định dạng gốc BF16:
     ```python
     return y.type_as(x).view(shape)
     ```
  - Kích thước tensor đầu ra: $[T, 4096]$ dạng BF16 $\to 2\text{ bytes/element} \times 4096 = 8\text{ KiB/token}$ ($2\text{ MiB}$ khi $T=256$).

---

## 5. Phân tích chi tiết các khối Ngoại vi (Ngoài ECA)

### Ngoài ECA · Post-block mHC (Sinkhorn Pre-mix & RMSNorm)
- **Vị trí code**: `model.py:673-681` (`hc_pre`), `model.py:659` (`ffn_norm`), `model.py:696-697`
- **Vai trò**:
  - Mô hình DeepSeek sử dụng **Hyper-Connections (HC)** với `hc_mult = 4`. Trạng thái ẩn được duy trì dưới dạng 4 luồng song song $X' \in \mathbb{R}^{B \times S \times 4 \times 4096}$.
  - Hàm `hc_pre` tính ma trận chuẩn hóa kép (doubly stochastic) qua thuật toán Sinkhorn (`hc_split_sinkhorn`), tạo ra hệ số `pre` để tổng hợp có trọng số 4 luồng $X'$ thành 1 luồng:
    $$y_{\text{pre}} = \sum_{k=0}^{3} \text{pre}_k \cdot X'_k \in \mathbb{R}^{B \times S \times 4096}$$
  - Tiếp theo, luồng này đi qua `ffn_norm` (RMSNorm với epsilon $10^{-6}$) để tạo ra đầu vào chuẩn xác $h [T, 4096]$ đưa vào Router và ECA.

### Ngoài ECA · MRU / Router (Gate)
- **Vị trí code**: `model.py:546-585` (Class `Gate`)
- **Vai trò**:
  - Nhận $h [T, 4096]$, tính toán điểm số cho 256 routed experts:
    $$\text{scores} = h \cdot W_{\text{gate}}^T$$
  - **Kích hoạt hàm điểm**: `score_func == "sqrtsoftplus"`:
    $$\text{scores} = \sqrt{\text{softplus}(\text{scores})}$$
  - **Layer 0-2 (Hash Layers)**: Với 3 layer đầu tiên (`layer_id < 3`), định tuyến được cố định trước thông qua bảng tra `self.tid2eid[input_ids]`.
  - **Layer 3-42 (Score Layers)**: Cộng thêm `bias`, sau đó chọn Top-6:
    $$\text{indices} = \text{topk}(\text{scores} + \text{bias}, k=6)$$
  - Chuẩn hóa trọng số của 6 chuyên gia được chọn về tổng bằng 1, sau đó nhân với `route_scale = 1.5`:
    $$\text{weights} = \frac{\text{original\_scores}[\text{indices}]}{\sum \text{original\_scores}[\text{indices}]} \times 1.5$$
  - Sinh ra $6T$ routed jobs cho các Engine của ECA.

### Ngoài ECA · WTB / DDR4 (Storage Core)
- **Vị trí code**: `model.py:108-152` (`Linear`), `kernel.py:204-276` (`fp8_gemm`), `kernel.py:442-537` (`fp4_gemm`)
- **Vai trò**:
  - **WTB (Weight Tile Buffer)** và bộ nhớ ngoài (DDR4 / HBM) quản lý lưu trữ toàn bộ tham số của 256 experts $\times 43$ layers.
  - Tổng dung lượng của toàn bộ chuyên gia vượt quá dung lượng SRAM on-chip, do đó chỉ có tile descriptors, scales và các khối trọng số được chọn mới được stream/fetch nạp vào ECA thông qua các kênh nạp chuyên biệt:
    - Nạp các **routed FP4 tiles** (khối $128 \times 32$ packed) cho Engine 0–3.
    - Duy trì **shared FP8 weights** ($24\text{ MiB}$) cho Shared Expert.

### Ngoài ECA · Residual Mix (MoE)
- **Vị trí code**: `model.py:683-686` (`hc_post`), `model.py:699`
- **Vai trò**:
  - Sau khi ECA hoàn tất và trả về $y [T, 4096]$, khối `hc_post` thực hiện hòa trộn trở lại vào 4 luồng của Hyper-Connections theo công thức:
    $$X_{\text{next}} = \text{post} \otimes y + \sum_{k=0}^{3} \text{comb} \otimes X'$$
  - Viết gọn theo đại số tuyến tính như trong sơ đồ Draw.io:
    $$\mathbf{X}_{\text{next}} = \mathbf{B} \cdot X' + \mathbf{C} \cdot y$$
    Trong đó:
    - $X' \in \mathbb{R}^{T \times 4 \times 4096}$ là residual bypass từ trước khối MoE.
    - $\mathbf{B}$ là ma trận tổ hợp $4 \times 4$ (`comb`).
    - $\mathbf{C}$ là vector phân phối $4$ chiều (`post`).
  - Đầu ra có kích thước $[T, 4, 4096]$ dạng BF16 ($32\text{ KiB/token}$), sẵn sàng đi tiếp vào Attention block tiếp theo.

---

## 6. Bảng đối chiếu & so sánh Spec (Draw.io) vs Thực tế Source Code

Dưới đây là bảng phân tích đối chiếu từng hạng mục giữa sơ đồ thiết kế Draw.io XML và mã nguồn triển khai thực tế trong repo:

| Hạng mục | Sơ đồ Draw.io Spec | Thực tế Source Code (`model.py` / `kernel.py`) | Đánh giá / Khớp hay Khác |
|:---|:---|:---|:---:|
| **Tên khối chính** | `ECA - TOP` | Khối bọc quanh `MoE` và `Expert` (`model.py:586, 608, 645`) | ✅ **Khớp 100%** (có chú thích `// ECA`) |
| **Số lượng Routed Experts** | 256 experts (6 per token) | `n_routed_experts=256`, `n_activated_experts=6` (`config.json:8,10`) | ✅ **Khớp 100%** |
| **Phân bổ Engine** | 4 Engines: 0 (0-63), 1 (64-127), 2 (128-191), 3 (192-255) | `n_local_experts = 256 // world_size = 64` (với `world_size=4`), `start_idx = rank * 64` | ✅ **Khớp 100%** |
| **Kích thước Ma trận Expert** | $W_1: [2048, 4096]$, $W_3: [2048, 4096]$, $W_2: [4096, 2048]$ | `w1: Linear(4096, 2048)` $\to$ weight $[2048, 4096]$;<br/>`w2: Linear(2048, 4096)` $\to$ weight $[4096, 2048]$ | ✅ **Khớp 100%** |
| **Định dạng Routed Expert** | `routed FP4 tiles` | `expert_dtype = torch.float4_e2m1fn_x2` (`model.py:623`) | ✅ **Khớp 100%** |
| **Định dạng Shared Expert** | `Official FP8-weight mode`, payload $24\text{ MiB}$ | `default_dtype = torch.float8_e4m3fn`; $3 \times (2048 \times 4096) = 24\text{ MiB}$ (`model.py:776`) | ✅ **Khớp 100%** |
| **Đặc điểm Shared Expert** | Mọi token, không route weight | `y += self.shared_experts(x)` (không truyền weights) | ✅ **Khớp 100%** |
| **Điểm nhân route_weight** | $W_1 \to W_3 \to \text{SwiGLU} \times \text{weight} \to W_2$ | `x = F.silu(gate) * up; if weights: x = weights * x; return w2(x)` | ✅ **Khớp 100%** |
| **Kiểu tích lũy Accumulation** | `Tagged FP32 accumulation across 6 slots` | `y = torch.zeros_like(x, dtype=torch.float32); y[idx] += expert(...)` | ✅ **Khớp 100%** |
| **Dung lượng $y_{\text{routed\_fp32}}$** | $[T, 4096]$ · $16\text{ KiB/token}$ | $4096 \times 4\text{ bytes (FP32)} = 16{,}384\text{ bytes} = 16\text{ KiB/token}$ | ✅ **Khớp 100%** |
| **Combine & Cast Output** | $y_{\text{full\_fp32}} \to y_{\text{bf16}} [T, 4096]$ ($8\text{ KiB/token}$, $2\text{ MiB}$ tại $T=256$) | `return y.type_as(x).view(shape)`; $4096 \times 2\text{ bytes} = 8\text{ KiB/token}$ | ✅ **Khớp 100%** |
| **Công thức Residual Mix** | $X_{\text{next}} = B \cdot X' + C \cdot y$ | `y = post * x + sum(comb * residual)` (`model.py:685`) | ✅ **Khớp 100%** |
| **Kích thước Bypass $X'$** | $[T, 4, 4096]$ BF16 ($32\text{ KiB/token}$, $8\text{ MiB}$ tại $T=256$) | $4 \times 4096 \times 2\text{ bytes} = 32\text{ KiB/token}$; $256 \times 32\text{ KiB} = 8\text{ MiB}$ | ✅ **Khớp 100%** |

---

### Các điểm khớp hoàn toàn (Exact Matches)
1. **Kiến trúc phân chia 4 Engine**: Mỗi engine đảm nhiệm đúng 64 experts.
2. **Kích thước tensor và bộ nhớ**: Tất cả các con số $8\text{ KiB/token}$, $16\text{ KiB/token}$, $32\text{ KiB/token}$, $2\text{ MiB}$, $8\text{ MiB}$, và $24\text{ MiB}$ đều khớp chính xác từng byte.
3. **Thứ tự nhân trọng số tuyến tính**: Nhân `weights` ngay sau SwiGLU trước khi nhân chiếu qua ma trận $W_2$.
4. **Bộ tích lũy FP32 Accumulator**: Toàn bộ quá trình cộng dồn 6 slot của routed experts và cộng dồn shared expert đều diễn ra trên kiểu số thực dấu phẩy động 32-bit (FP32) để tránh trôi dạt số học (numerical drift).

---

### Các điểm khác biệt & chi tiết cần lưu ý đặc biệt (Discrepancies & Nuances)

Dưới đây là các chi tiết thực tế trong source code mà sơ đồ thiết kế Draw.io đã trừu tượng hóa:

#### 1. Động lượng phân tán: Hardware Engine vs PyTorch Tensor Parallelism
- **Trong sơ đồ**: Thể hiện 4 Engine (0, 1, 2, 3) như 4 khối tính toán phần cứng song song nằm trên cùng 1 die/chip ECA.
- **Trong source code**: Code được viết theo mô hình phân tán **Tensor Parallelism (TP)** của PyTorch. Mỗi Engine tương ứng với một `rank` GPU riêng biệt:
  ```python
  self.n_local_experts = args.n_routed_experts // world_size # 64 khi world_size=4
  self.experts_start_idx = rank * self.n_local_experts
  self.experts_end_idx = self.experts_start_idx + self.n_local_experts
  ```
  Khi kết thúc vòng lặp tính toán cục bộ, code gọi:
  ```python
  if world_size > 1:
      dist.all_reduce(y) # Đồng bộ All-Reduce tổng qua giao tiếp liên GPU (NVLink)
  ```
  Nếu chạy ở chế độ đơn GPU (`world_size = 1`), toàn bộ 256 experts sẽ được chạy tuần tự trên Engine duy nhất.

#### 2. Giới hạn tràn số SwiGLU (`swiglu_limit = 10.0`)
- **Trong sơ đồ**: Không đề cập đến bước clamping này.
- **Trong code**: `model.py:600-602` có logic:
  ```python
  if self.swiglu_limit > 0:
      up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
      gate = torch.clamp(gate, max=self.swiglu_limit)
  ```
  Trong `config.json`, giá trị này được cấu hình `"swiglu_limit": 10.0`. Đây là cơ chế bảo vệ cực kỳ quan trọng để ngăn hiện tượng bùng nổ gradient và tràn số khi nhân với ma trận lượng tử hóa FP4/FP8.

#### 3. Định lượng động đầu vào (Dynamic Activation Quantization - `act_quant`)
- **Trong sơ đồ**: Vẽ mũi tên trực tiếp mang tensor $h [T, 4096]$ BF16 vào Engine.
- **Trong code**: Vì ma trận trọng số $W_1, W_2, W_3$ được lưu ở định dạng FP4 và FP8, trước khi nhân GEMM, đầu vào activation BF16 **bắt buộc phải qua hàm `act_quant`** (`model.py:114, 117` và `kernel.py:171-200`):
  ```python
  x, s = act_quant(x, block_size=128, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
  ```
  Nghĩa là tensor $h$ BF16 được chuyển đổi động on-the-fly sang FP8 kèm hệ số tỷ lệ `s` theo từng khối 128 phần tử trên chiều $K$. Kernel TileLang `fp4_gemm_kernel` sau đó thực hiện nhân:
  $$\text{Act (FP8)} \times \text{Weight (FP4)} \to \text{Output (BF16/FP32)}$$

#### 4. Phân loại Layer: Hash Routing (Lớp 0-2) vs Score Routing (Lớp 3-42)
- **Trong sơ đồ**: Khối Router chỉ mô tả tính toán điểm số chung (`route_indices`, `route_weights`).
- **Trong code**: `config.json` định nghĩa `"n_hash_layers": 3`.
  - Tại 3 layer đầu tiên (`layer_id < 3`), router không tính toán tích vô hướng $h \cdot W^T$ mà tra cứu trực tiếp từ token ID qua bảng `self.tid2eid[input_ids]`.
  - Từ layer 3 trở đi mới thực hiện tính điểm qua hàm `sqrtsoftplus` và cộng vector `bias`.

#### 5. Điều phối Token: GPU Sequential Loop vs HW Pipeline Dispatch
- **Trong sơ đồ**: Thể hiện như một pipeline phần cứng tự động phân luồng đồng thời 6 slot vào 4 Engine.
- **Trong code**: Triển khai PyTorch giả lập hành vi này bằng vòng lặp `for` lặp qua 64 experts cục bộ, sử dụng `torch.bincount` để lọc bỏ các expert không có token nào chọn (`counts[i] == 0`), và dùng `torch.where(indices == i)` để lấy ra các token tương ứng.

---

## 7. Hướng dẫn đọc code và thực thi từng bước (Step-by-Step Code Walkthrough)

Để nắm bắt toàn bộ luồng hoạt động trong mã nguồn, hãy làm theo quy trình 4 bước sau:

### Bước 1: Khởi nguồn từ Transformer Block (`model.py:695-700`)
```python
# 1.1 Lưu trữ bypass 4 luồng
residual = x  # x có kích thước [b, s, 4, 4096]

# 1.2 Thu gọn 4 luồng thành 1 luồng bằng Sinkhorn weights
x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)

# 1.3 Chuẩn hóa RMSNorm để ra h [T, 4096]
x = self.ffn_norm(x)

# 1.4 Đi vào khối MoE (ECA - TOP)
x = self.ffn(x, input_ids)

# 1.5 Hòa trộn trở lại 4 luồng Hyper-Connections
x = self.hc_post(x, residual, post, comb)
```

### Bước 2: Bộ định tuyến Router chọn Top-6 (`model.py:564-584`)
```python
scores = linear(x.float(), self.weight.float())
scores = F.softplus(scores).sqrt() # score_func = sqrtsoftplus
if self.bias is not None:
    scores = scores + self.bias
indices = scores.topk(self.topk, dim=-1)[1] # Top 6 expert IDs
weights = original_scores.gather(1, indices)
weights /= weights.sum(dim=-1, keepdim=True) # Chuẩn hóa L1
weights *= self.route_scale                  # Nhân hệ số 1.5
```

### Bước 3: Lõi tính toán ECA - TOP (`model.py:630-645`)
```python
shape = x.size()
x = x.view(-1, self.dim) # Flatten [T, 4096]
weights, indices = self.gate(x, input_ids.flatten())

# Khởi tạo bộ đệm tích lũy Tagged FP32 Accumulator
y = torch.zeros_like(x, dtype=torch.float32)

counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

# Expert-major loop chạy trên 64 experts cục bộ của Engine/Rank hiện tại
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)
    # Tính SwiGLU x route_weight và tích lũy FP32
    y[idx] += expert(x[idx], weights[idx, top, None])

# Gom kết quả từ 4 Engine nếu chạy đa thiết bị
if world_size > 1:
    dist.all_reduce(y)

# Khối B: Tính toán Shared Expert cho mọi token và cộng dồn
y += self.shared_experts(x)

# Khối C: Cast ngược lại BF16 và reshape về cấu trúc ban đầu
return y.type_as(x).view(shape)
```

### Bước 4: Lõi tính toán bên trong từng Expert (`model.py:596-606`)
```python
dtype = x.dtype # BF16
gate = self.w1(x).float() # Linear FP4 -> Gate [B_sub, 2048] FP32
up = self.w3(x).float()   # Linear FP4 -> Up   [B_sub, 2048] FP32

# Kẹp giá trị chống tràn số
if self.swiglu_limit > 0:
    up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
    gate = torch.clamp(gate, max=self.swiglu_limit)

x = F.silu(gate) * up # SwiGLU activation

# Nhân trọng số định tuyến trước khi down-projection (tiết kiệm FLOPs)
if weights is not None:
    x = weights * x

# Chiếu ngược lại không gian 4096 qua W2
return self.w2(x.to(dtype))
```

---

## TỔNG KẾT
Toàn bộ cụm **ECA - TOP** trong file thiết kế Draw.io XML khớp một cách hoàn hảo với logic triển khai mã nguồn tại `model.py` và `kernel.py`:
- **Về mặt kích thước & tensor**: Tất cả các chiều $[T, 4096]$, $[2048, 4096]$, $[4096, 2048]$, dung lượng bộ nhớ ($8\text{ KiB}$, $16\text{ KiB}$, $32\text{ KiB}$, $2\text{ MiB}$, $8\text{ MiB}$, $24\text{ MiB}$) đều chính xác 100%.
- **Về mặt kiến trúc**: Điểm cốt lõi nhất nằm ở việc **tách biệt 256 routed experts (FP4) cho 4 Engine**, **chia sẻ 1 shared expert (FP8)**, **nhân route weight trước ma trận $W_2$**, và duy trì **tích lũy Tagged FP32 Accumulator** để đảm bảo độ chính xác cao nhất trước khi trả về BF16.
