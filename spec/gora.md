# spec/gora.md — Đặc tả GoRA (thành phần `O` + `Oi`) cho lora-stack-lab

> Clean-room: viết từ paper **GoRA: Gradient-driven Adaptive Low Rank Adaptation** (arXiv 2502.12171, NeurIPS 2025), không đọc mã tác giả.
> Nguồn công thức: §3.1–3.3, Algorithm 1, Appendix A.1–A.2, B.3 của paper (fetch ar5iv ngày 2026-11-01).
> Quy ước ghi nguồn: `[paper, Eq.(n)]` = phát biểu của paper; `[quyết]` = quyết định thiết kế của dự án; `[paper-ambiguity]` = chỗ paper không rõ, xử lý tại mục 9.

---

## 1. Mục tiêu & phạm vi

Luận điểm gốc `[paper, §3.1]`: adapter LoRA là một **bộ nén gradient (gradient compressor)**. Trước train, tính gradient trung bình $G = \frac{1}{N}\sum_{i=1}^{N}\frac{\partial L_i}{\partial W}$ trên $N$ batch; dùng $G$ để (a) cấp rank theo độ quan trọng, (b) khởi tạo $B$ sao cho $A\cdot B$ xấp xỉ tốt nhất phép chiếu của $G$.

Phạm vi dự án — tách 2 thành phần độc lập, bật/tắt riêng (khác định nghĩa rút gọn trong plan.md mục 1, theo chỉ đạo điều phối):

| Mã | Nội dung | Hook registry | Chạy lúc |
|----|----------|---------------|----------|
| `O` | **Chỉ cấp rank**: tính $G$ streaming → importance → cấp $r_i$ từng layer | `allocate_ranks` + `patch_config` | trước `get_peft_model` |
| `Oi` | **Chỉ init**: $B_0 = -\xi\,\hat A^{-1}$-dạng giả nghịch đảo của $G$ | `init_adapter` | sau `get_peft_model` |

- `O` mà không `Oi`: adapter vẫn init kiểu PEFT (A kaiming, B = 0), chỉ khác rank từng layer.
- `Oi` mà không `O`: mọi layer dùng chung $r = r_{\text{ref}}$, nhưng $B$ được init theo gradient.
- GoRA **không đụng $W_0$** `[paper, §2.2, Appendix C.1]` → không xung đột X1 với PiSSA (`ctx.init_mode = "gora"` khi `Oi` bật, `ctx.init_mode` giữ nguyên khi chỉ `O`).

Ngoài phạm vi: learning-rate ratio $B{:}A = 16{:}1$ `[paper, B.3]` thuộc miền optimizer (component `A`/`L`), ghi ở mục 9.

---

## 2. Ký hiệu & shapes

### 2.1 Quy ước paper vs PEFT (bắt buộc phân biệt)

Paper `$[paper, §1]`: $W\in\mathbb{R}^{m\times n}$, $A\in\mathbb{R}^{m\times r}$, $B\in\mathbb{R}^{r\times n}$, $\Delta W = s\, A B$.

PEFT v0.21.2 (`LoraLayer` cho `nn.Linear` có weight $(m, n)$ = `(out_features, in_features)`):
- `lora_A.weight`: $(r, n)$ — n = in_features
- `lora_B.weight`: $(m, r)$ — m = out_features
- forward: $\Delta W = s\cdot\,\texttt{lora\_B}\,@\,\texttt{lora\_A}$, với $s = \alpha/r$ (mặc định) hoặc $\alpha/\sqrt r$ (khi `use_rslora=True`).

**Bảng ánh xạ** (dùng xuyên suốt spec):

| Paper | PEFT | Shape |
|-------|------|-------|
| $A$ (yếu tố random, kaiming) | `lora_B.weight` | $(m, r)$ |
| $B$ (yếu tố tính từ $G$) | `lora_A.weight` | $(r, n)$ |
| $G$ | gradient của `base_layer.weight` | $(m, n)$ |

$G$ luôn là gradient của weight $(m,n)$, không đổi theo quy ước.

### 2.2 Ký hiệu dùng trong spec

- Tập layer mục tiêu: $\{W_i\}_{i=1}^{L}$, $W_i \in \mathbb{R}^{m_i\times n_i}$.
- $r_{\text{ref}}$ = `cfg.rank`; ngân sách tham số LoRA thường: $B_{\text{tot}} = \sum_i (m_i+n_i)\,r_{\text{ref}}$.
- $N$ = `cfg.gora_n_batches` (paper: 64).
- $k$ = `cfg.gralora_k` (chỉ dùng cho tổ hợp X2 = `G+O`).

---

## 3. Công thức đầy đủ

### 3.1 Gradient tích lũy

$$G_i = \frac{1}{N}\sum_{t=1}^{N}\frac{\partial \ell_t}{\partial W_i} \qquad [paper, §3.2;\ \text{Algorithm 1, dòng 4–10}]$$

Mỗi bước $t$: lấy 1 mini-batch **ngẫu nhiên** từ tập train, forward + backward, **không optimizer step**, cộng grad vào buffer CPU rồi giải phóng GPU ngay (Algorithm 1 dòng 10: *"offload to cpu without optimizer step"*).

Tính chất hữu ích: vì importance sẽ bị **chuẩn hoá** ở (2), mọi hằng số scale toàn cục trên $G$ (kể cả loss scale của `GradScaler`) tự triệt tiêu — chỉ cần tỉ lệ **giữa các layer** đúng.

### 3.2 Importance & advantage

$$\mathrm{I}(W_i) = \mathrm{avg}\big(|W_i \odot G_i|\big) \qquad [paper, Eq.(5)]$$
$$A_i = \frac{\mathrm{I}(W_i)}{\sum_{j=1}^{L} \mathrm{I}(W_j)} \qquad [paper, Eq.(6)]$$

$\odot$: nhân từng phần tử; $\mathrm{avg}(\cdot)$: trung bình toàn bộ phần tử → 1 scalar/layer. (Paper đã ablate: metric này thắng nuclear norm của $G$ và của $W\odot G$ — Table 10 — nên **không** cài nuclear norm.)

### 3.3 Ngân sách & cấp rank

Ngân sách một layer dưới LoRA thường: $(m_i+n_i)\cdot r_{\text{ref}}$; tổng: $B_{\text{tot}}=\sum_i (m_i+n_i) r_{\text{ref}}$ `[paper, Eq.(7) "reflecting the expected parameter cost under LoRA"]`.

Phân bổ `[paper, Eq.(8) + Algorithm 1 dòng 17]`:

$$r_i = \mathrm{clip}\!\left(\mathrm{round}\!\left(\frac{B_{\text{tot}}\cdot A_i}{m_i+n_i}\right),\ r_{\min},\ r_{\max}\right)$$

- `[·]` = round tới số nguyên gần nhất `[paper, Eq.(8)]`.
- $r_{\min} = r_{\text{ref}}/2$, $r_{\max} = 4\,r_{\text{ref}}$ `[paper, B.3]`.
- Dạng proportional + clamp hai phía + round, **không** có bước cân bằng lại sau clip (xem mục 9.7).

**[paper-ambiguity]** Bản render ar5iv của Eq.(7)–(8) viết $B_i = \sqrt{m+n}\cdot r_{\text{ref}}$ và $r_i = [B\cdot A_i/\sqrt{m+n}]$. Dạng này mâu thuẫn với (i) chính câu "reflecting the expected parameter cost under LoRA" (chi phí LoRA là $(m+n)r_{\text{ref}}$) và (ii) Table 11 (GoRA 7.00M vs LoRA 6.82M tham số — chênh ~3%; nếu có $\sqrt{\cdot}$ thì lệch bậc $\sqrt{m+n}\approx 64$ lần với $d=4096$). Kết luận: dùng $(m+n)$, không căn. Chi tiết mục 9.1.

Tính chất: nếu mọi layer cùng shape và không clamp thì $\sum_i (m_i+n_i) r_i = B_{\text{tot}}\sum A_i = B_{\text{tot}}$ — khớp ngân sách LoRA đúng bằng cấu trúc. Clamp gây trôi, cam kết paper "within 10%" `[paper, §3.2 mục tiêu (3)]`.

### 3.4 Init $B_0$ (thành phần `Oi`)

Mục tiêu `[paper, Eq.(9)]`: với $A$ cố định (random kiểu LoRA), chọn $B$ sao cho $AB$ là xấp xỉ hạng-$r$ **tối ưu theo Frobenius** của $G$:

$$B = -(A^{\mathsf T}A)^{-1}A^{\mathsf T}G,\qquad AB = -P_A\,G,\quad P_A = A(A^{\mathsf T}A)^{-1}A^{\mathsf T}$$

$P_A$ là máy chiếu lên không gian cột của $A$; chứng minh tối ưu ở Appendix A.1. Tương đương Moore–Penrose: $B = -A^{+}G$.

**Dịch sang quy ước PEFT** (giữ `lora_A` kaiming — tức "A₀ giữ init kiểu LoRA"; tính `lora_B`):

$$\texttt{lora\_B}_0 = -\,\xi\cdot G\,M^{\mathsf T}\big(M M^{\mathsf T} + \varepsilon I_r\big)^{-1},\qquad M = \texttt{lora\_A}_0 \in \mathbb{R}^{r\times n}$$

Đây là dạng chuyển vị của Eq.(9): $G M^{\mathsf T}(MM^{\mathsf T})^{-1}$ = nghiệm least-squares tối ưu $\min_{B}\|G - B M\|_F$; $B\,M = -G P_{\text{row}}$, $P_{\text{row}} = M^{\mathsf T}(MM^{\mathsf T})^{-1}M$ là máy chiếu lên không gian hàng của $M$. Nghịch đảo cần là $r\times r$ (nhỏ), không bao giờ $n\times n$.

### 3.5 Scaling factor $\xi$

Paper hiệu chỉnh để $\Delta W_{\text{init}} = s\cdot \texttt{lora\_B}_0\,\texttt{lora\_A}_0 \approx -\gamma G$ — tương đương **một bước SGD với step $\gamma$** `[paper, Eq.(10)]`.

Từ Appendix A.2: với $G$ gaussian i.i.d., $\mathbb{E}\|P G\|_F = \sqrt{nr}$, $\mathbb{E}\|G\|_F = \sqrt{mn}$ → tỉ số $\sqrt{r/m}$. Quy ra PEFT (máy chiếu phía hàng, thay $m\leftrightarrow n$): $\mathbb{E}\|G P_{\text{row}}\|_F = \sqrt{mr}$, tỉ số $\sqrt{r/n}$.

| Quy ước init | Hệ số tỉ lệ ||  $\Delta W_{\text{init}}\|/\|G\|$ | $\xi$ với $s=\alpha/r$ | $\xi$ với $s=\alpha/\sqrt r$ |
|---|---|---|---|
| Paper-literal (random $A$ cỡ $m\times r$ = `lora_B`) | $\sqrt{r/m}$ | $\gamma\sqrt{rm}/\alpha$ `[paper, §3.3]` | $\gamma\sqrt m/\alpha$ `[paper, Alg.1 dòng 20]` |
| PEFT-native (random `lora_A` cỡ $r\times n$) | $\sqrt{r/n}$ | $\gamma\sqrt{rn}/\alpha$ | $\gamma\sqrt n/\alpha$ |

Với attention $q,k,v,o$: $m=n=d$ → hai quy ước cho cùng $\xi$. Với MLP `gate/up` ($m>n$) và `down` ($m<n$) thì khác — phải dùng đúng cột của bảng theo quy ước đã chọn. **[quyết]** mặc định PEFT-native (giữ nguyên init `lora_A` của PEFT, `use_rslora` theo `peft_config`); paper-literal là biến thể flag (mục 9.8).

GoRA đầy đủ trong paper dùng forward scaling $\alpha/\sqrt r$ (kiểu RSLoRA) `[paper, §3.3]` — xem mục 6.4 và 9.9 (CAO).

---

## 4. Giả mã thuật toán (2 phase)

### Phase (a) — streaming tính $G$ (hook `allocate_ranks`, trước `get_peft_model`)

```text
# Trên model BASE, chưa có adapter (pipeline: O -> build -> init).
# Bật grad trên weight mục tiêu (mặc định frozen):
for name, p in named_target_params(model):      # khớp cfg.target_modules
    p.requires_grad_(True)
    p.register_hook(hook_accum(name))           # grad (GPU, fp16/fp32)
G[name] = torch.zeros_like(p, dtype=fp32, device="cpu")   # buffer CPU

set_seed(cfg.seed)                              # tái lập (plan.md quy tắc 5)
loader_G = DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
for t in range(N):                              # N = cfg.gora_n_batches = 64
    x, y = next(loader_G)
    loss = CE(model(x), y)                      # fp32, KHÔNG autocast [quyết, mục 7]
    loss.backward()                             # hook tự cộng; KHÔNG optimizer.step()
    model.zero_grad(set_to_none=False)

for name, p in named_target_params(model):
    p.requires_grad_(False)
del hook; model.zero_grad(set_to_none=True)     # giải phóng GPU ngay
# G[name] giữ trên CPU fp32.
```

`hook_accum(name)`: `G[name] += g.detach().float().cpu()` (grad per-batch, hệ $1/N$ gộp ở cuối hoặc cộng dồn rồi chia — kết quả như nhau).

### Phase (b) — cấp rank (cùng hook `allocate_ranks`, trả dict)

```text
for i: I[i] = avg(|W_i.float() * G[name_i]|)          # Eq.(5)
A[i]  = I[i] / sum(I)                                  # Eq.(6)
B_tot = Σ (m_i + n_i) * r_ref                          # Eq.(7), dạng (m+n)
for i:
    r_raw = round(B_tot * A[i] / (m_i + n_i))
    r[i]  = clip(r_raw, r_min, r_max)                  # r_min = r_ref//2, r_max = 4*r_ref
# X2 (G+O): r[i] = quy về bội của k (mục 6.3), ghi lệch ngân sách
patch_config: peft_config.rank_pattern[full_name_i] = r[i]   # từng layer
return {full_name_i: r[i]}                # -> ctx.layer_ranks
```

### Phase (c) — init $B_0$ (hook `init_adapter`, sau `get_peft_model`, dùng `ctx.ab_pairs`)

```text
# ctx.gora_G: dict[name -> G fp32 CPU] do phase (a) để lại trên ctx
for (name, lora_A, lora_B) in ctx.ab_pairs:
    G   = ctx.gora_G[key_of(name)].to(fp32, device)     # lên GPU 1 layer/lần
    M   = lora_A.weight.detach().float()                # (r, n), giữ kaiming của PEFT
    H   = M @ M.T                                       # (r, r)
    H  += eps_rel * trace(H)/r * I_r                    # ridge, eps_rel = 1e-6 [quyết]
    R   = G @ M.T                                       # (m, r)
    B0  = -solve(H, R.T).T                              # (m, r) = -xi * G M^T (MM^T+eI)^-1
    xi  = xi_table(scaling_mode, gamma, m, n, r, alpha) # bảng mục 3.5
    B0 *= xi
    with no_grad(): lora_B.weight.copy_(B0.to(lora_B.weight.dtype))  # fp16/fp32 theo param
    del G, B0                                           # giải phóng ngay
ctx.init_mode = "gora"
```

Thứ tự pipeline_consumes: `allocate_ranks` → `patch_config` → build (`L|G`) → `init_adapter` — đúng comment `registry.py` dòng 2–3.

---

## 5. Siêu tham số mặc định (từ paper)

| Tham số | Giá trị paper | Nguồn | Trong `config.py` hiện tại |
|---|---|---|---|
| $N$ (số batch tính $G$) | 64 | B.3, Alg.1 | `gora_n_batches=64` ✓ |
| $r_{\min}$ | $r_{\text{ref}}/2$ | B.3 | `gora_r_min=None` → default `rank//2` ✓ |
| $r_{\max}$ | $4\,r_{\text{ref}}$ | B.3 | **thiếu** — cần `gora_r_max` |
| $\gamma$ | r₀=8: 5e-2 (MetaMathQA: 8e-2); r₀=32: 1e-2; r₀=128: 5e-3 | B.3, Table 9 | **thiếu** — cần `gora_gamma` |
| $\varepsilon$ (ridge) | paper không nêu | — | mặc định $\varepsilon_{\text{rel}}=10^{-6}$ `[quyết]` |
| batch khi tính $G$ | = batch size training (64 với Llama3.1) | Alg.1 dòng 6 + §4.2 | dùng `cfg.batch_size` `[quyết]` |
| lr ratio $B{:}A$ | 16 : 1 | B.3 | thuộc optimizer, ngoài `O/Oi` (mục 9.10) |
| scaling forward | $\alpha/\sqrt r$ (kiểu RSLoRA) | §3.3 | cần flag `gora_use_rslora` (mục 9.9) |
| seed | paper: 3 seeds, không nêu seed riêng cho init | §4 | `cfg.seed`, set trước phase (a) và trước `get_peft_model` `[quyết]` |
| dtype $G$ | không nêu (train BF16/FP32) | — | buffer CPU fp32 `[quyết]` |

Thời gian phase (a): paper ~4 phút cho 64 bước trên 8B/4090 `[paper, Table 12]` — với Qwen 0.5B trên T4 không đáng kể so với TimeGuard.

---

## 6. Điểm tích hợp (theo `registry.py` / `base.py`)

### 6.1 Hook `allocate_ranks(cfg, model, tok, loader) -> dict[layer_name, int]` (O)

- Thực hiện phase (a) + (b). Đăng ký bởi module `lorastack.adapters.gora_alloc` (mapping sẵn trong `registry.py` dòng 30).
- Đầu vào `loader`: DataLoader train đã shuffle; dùng `cfg.batch_size`.
- Trả dict **tên module đầy đủ không prefix `base_model.model.`** (ví dụ `model.layers.5.self_attn.q_proj`) → $r_i$; pipeline ghi vào `ctx.layer_ranks`.
- Đăng ký thêm hook `patch_config(cfg, peft_config)`: nạp `peft_config.rank_pattern[name] = r_i` (PEFT 0.21.2 khớp key kiểu suffix-regex `.*\.{key}$` — cần smoke test 1 layer, mục 9.11).

### 6.2 Hook `init_adapter(cfg, model, ctx) -> None` (Oi)

- Đăng ký bởi `lorastack.adapters.gora_init`.
- Điều kiện chạy: `ctx.gora_G` tồn tại trên ctx (dataclass cho phép gắn attr động) **hoặc** nhận $G$ qua file tạm `results/<exp_id>/gora_G.pt` nếu O và Oi chạy ở tiến trình khác nhau — mặc định truyền qua ctx `[quyết]`.
- Duyệt `ctx.ab_pairs` (đã có sau `get_peft_model`), thực hiện phase (c), set `ctx.init_mode = "gora"`.

### 6.3 X2 (`G+O`): quy $r_i$ về bội của `gralora_k`

GraLoRA yêu cầu rank chia hết cho $k$ (cấu trúc lưới khối). Sau Eq.(8):

$$r_i' = k\cdot\mathrm{clip}\!\left(\mathrm{round}\!\left(\tfrac{r_i}{k}\right),\ \lceil r_{\min}/k\rceil,\ \lfloor r_{\max}/k\rfloor\right)$$

Lệch ngân sách so với `O` thuần:

$$\text{dev} = \sum_i (m_i+n_i)(r_i' - r_i),\qquad |\text{dev}| \le \tfrac{k}{2}\sum_i (m_i+n_i) = \tfrac{k}{2}\cdot\frac{B_{\text{tot}}}{r_{\text{ref}}}$$

Tương đối: $|\text{dev}|/B_{\text{tot}} \le k/(2 r_{\text{ref}})$ → với $k=2, r_{\text{ref}}=32$: ≤ ~3,1%. Ghi `ctx.rank_budget_deviation = dev` (signed, int) — JSON kết quả ghi cả `%` so $B_{\text{tot}}$ (field đã có sẵn trong `base.py` dòng 17).

### 6.4 Tổ hợp và scaling

- `Oi` đơn lẻ (`L+Oi`) hoặc full GoRA (`L+O+Oi`): paper dùng forward scaling $\alpha/\sqrt r$. Đề xuất: khi `Oi ∈ components`, pipeline dựng `LoraConfig(use_rslora=True)` — **nhưng L là đối chứng đã tune LR, đổi scaling của L là đụng đối chứng** → bắt buộc hỏi user (mục 9.9, CAO).
- `G+Oi` (TRIAD): GraLoRA có cơ chế scaling riêng → $\xi$ phải tính theo **scaling thực tế của adapter được build**, đọc từ `peft_config` sau build (không hardcode).

---

## 7. fp16 / T4

1. **Phase (a) chạy fp32, không autocast** `[quyết]`: grad fp16 tích lũy 64 bước có rủi ro underflow ở phần tử nhỏ, làm méo tỉ lệ importance giữa các layer (phần nào mất thông tin không hồi phục được bằng chuẩn hoá). Qwen2.5-0.5B fp32 (~2 GB weights + 2 GB grads + activation) vừa T4 16 GB với `cfg.batch_size`, bật `gradient_checkpointing` nếu cần.
2. **Mọi phép đại số tuyến tính của `Oi` (giao `M@Mᵀ`, solve, `G@Mᵀ`) thực hiện fp32 trên GPU**, sau đó `copy_` xuống param theo dtype thật của param (plan.md §2: adapter giữ fp32 trên T4 → thường là fp32; nếu param fp16 thì cast khi copy, sai số ≤ eps fp16, chấp nhận).
3. Ridge $\varepsilon_{\text{rel}} = 10^{-6}\cdot\mathrm{tr}(MM^{\mathsf T})/r$: $MM^{\mathsf T}$ của ma trận kaiming $(r\times n)$ với $r\le n$ khả nghịch gần như chắc chắn, nhưng fp16-tồn-dư hoặc $r$ lớn có thể làm điều số xấu — ridge là bắt buộc, không optional.
4. Không giữ toàn bộ $G$ trên GPU cùng lúc: buffer CPU fp32, mỗi layer lên GPU 1 lần rồi `del`.
5. Sau phase (a): `zero_grad(set_to_none=True)`, trả `requires_grad_(False)` cho weight mục tiêu — không để sóng grad làm optimizer của bước train sau nhặt phải.

---

## 8. Tests tính chất đề xuất (1 lớp `nn.Linear`, không cần mã gốc)

| # | Tính chất | Cách kiểm |
|---|-----------|-----------|
| T1 | $M M^{\mathsf T}$ khả nghịch với $A_0$ kaiming, $r \le n$ | `cond(M@M.T)` hữu hạn (< 1e8 sau ridge) |
| T2 | $A_0 B_0 \approx -P_{\text{row}} G$ với $G$ tổng quát hạng thấp khả nghịch | tạo $G = U V^{\mathsf T}$ ($U$ $m\times r$, $V$ $n\times r$); assert $\|BM + G P_{\text{row}}\|_F / \|G P_{\text{row}}\|_F < 10^{-4}$ (fp32, ridge không tác động khi điều kiện số tốt) |
| T3 | Ngân sách sau round+clip | $\big|\sum (m_i+n_i) r_i - B_{\text{tot}}\big| / B_{\text{tot}} \le 10\%$ trên cấu hình shape thật của Qwen2.5-0.5B |
| T4 | Không có $r < r_{\min}$, không có $r > r_{\max}$ | assert trực tiếp trên dict |
| T5 | $k=1$ + importance đều → khớp LoRA thường | mọi $I_i$ bằng nhau ⇒ mọi $r_i = r_{\text{ref}}$ (khi shape bằng nhau, không clamp) |
| T6 | Monotonicity | cùng $(m+n)$: $I_i > I_j$ ⇒ $r_i \ge r_j$ |
| T7 | Streaming = one-shot | $G$ từ hook 64 bước ≡ autograd.grad tích lũy thủ công trên cùng 64 batch, sai lệch tương đối < 1e-5 (fp32) |
| T8 | $\xi$ đúng chuẩn | $q\_proj$ ($m=n$): $\|\Delta W_{\text{init}}\|_F / \|G\|_F \approx \gamma$ (sai số 20% do hội tụ random-matrix) |
| T9 | X2 rounding | mọi $r_i' \bmod k = 0$; `rank_budget_deviation` đúng bằng $\sum(m_i+n_i)(r_i'-r_i)$; tỉ lệ ≤ $k/(2r_{\text{ref}})$ |
| T10 | dtype | solve chạy fp32 (whitebox assert dtype input); giá trị trong param đích bằng `B0.to(dtype_param)` |

---

## 9. Điểm CHƯA XÁC ĐỊNH (paper không nêu rõ) — kèm xử lý đề xuất + mức ảnh hưởng

| # | Vấn đề | Đề xuất xử lý | Ảnh hưởng |
|---|--------|---------------|-----------|
| 9.1 | Eq.(7)–(8) render $\sqrt{m+n}$ nhưng mâu thuẫn "expected parameter cost under LoRA" và Table 11 | Dùng $(m+n)$ (đã chứng minh bằng nhất quán nội tại của paper). Nếu muốn khớp 100% mã gốc, nhờ user xác nhận từ mã tác giả | **vừa** (đã tự xử lý có bằng chứng; rủi ro còn lại nhỏ) |
| 9.2 | $\gamma = 0$ trong Table 9 nghĩa gì: $\xi = 0$ (B = 0, trông như LoRA nhưng điểm khác hẳn) hay $\xi = 1$ (bỏ hiệu chỉnh, giữ scale tự nhiên của giả nghịch đảo) | Diễn giải $\xi = 1$ (điểm 72.45 ≠ 67.78 của LoRA). Không dùng $\gamma=0$ trong thí nghiệm của dự án | thấp |
| 9.3 | $\gamma$ chỉ công bố cho $r_{\text{ref}} \in \{8, 32, 128\}$; dự án dùng $r=32$ (có giá trị 1e-2) nhưng sweep LR có thể dùng rank khác | Bảng 3 mức + nội suy tuyến tính theo $1/\sqrt{r_{\text{ref}}}$ cho rank khác; ghi rõ vào JSON | thấp–vừa |
| 9.4 | Trạng thái model khi tính $G$: train hay eval (dropout on/off); adapter đã gắn chưa (paper gắn với $B=0$ — tương đương base) | Pipeline của dự án chạy O **trước khi build adapter** → thuần base, không mơ hồ về adapter; mode: `model.train()` + seed cố định (khớp phân phối loss training) | thấp |
| 9.5 | $G$ tính trên batch đầu tiên hay batches ngẫu nhiên cố định; shuffle riêng? | Alg.1 dòng 6: "Randomly sampled mini-batch" mỗi bước → DataLoader shuffle với seed `cfg.seed`, 64 bước đầu của chính loader train | thấp |
| 9.6 | dtype của $G$ khi train mixed precision | Buffer CPU fp32, phase (a) chạy fp32 (mục 7) | thấp |
| 9.7 | Không có cơ chế bù ngân sách khi nhiều layer chạm clamp ($r_{\min}$/$r_{\max}$) — tổng params có thể trôi >10% | Sau round: nếu lệch >10%, scale $B_{\text{tot}} \leftarrow B_{\text{tot}}\cdot B_{\text{tot}}/\hat P$ và round lại **một** vòng (deterministic); vẫn lệch thì ghi cảnh báo JSON, không lặp | vừa |
| 9.8 | Hướng yếu tố random: paper kaiming ma trận $m\times r$ (= `lora_B` theo PEFT), PEFT mặc định kaiming $r\times n$ (= `lora_A`). Hai dạng cùng tối ưu Frobenius nhưng phân phối kaiming khác (fan_in khác) | Mặc định PEFT-native (giữ `lora_A` nguyên vẹn — đúng tinh thần "A₀ giữ init kiểu LoRA" trong hệ PEFT); flag `gora_paper_literal_init` để đổi sang paper-literal nếu user muốn khớp mã gốc | vừa |
| 9.9 | GoRA đầy đủ dùng forward scaling $\alpha/\sqrt r$ (kiểu RSLoRA) như một phần của phương pháp; nhưng trong dự án, `L` là đối chứng đã tune LR theo $\alpha/r$ — bật `use_rslora` cho combo có `Oi` sẽ thay đổi luôn baseline | **CAO — cần điều phối hỏi user**: (a) `Oi` đi kèm `use_rslora=True` (khớp paper, nhưng khi so `L+Oi` vs `L` thì khác 2 trục cùng lúc), hay (b) giữ $\alpha/r$ và chỉ đổi công thức $\xi$ theo bảng mục 3.5 (so sánh sạch, lệch paper) | **CAO** |
| 9.10 | lr ratio $B{:}A = 16{:}1$ (B.3) thuộc optimizer; `config.py` chưa có field; không áp dụng sẽ lệch với paper khi tái lập GoRA đầy đủ | Thêm `gora_lr_ratio` — nhưng đây là miền component `A`/`L`; để điều phối quyết định đặt ở đâu | vừa |
| 9.11 | Semantics `rank_pattern` PEFT 0.21.2 (khớp suffix regex `.*\.{key}$`): key tên đầy đủ có khớp cho mọi layer không (đặc biệt module ngay dưới gốc) | Smoke test 1 model trước khi chạy thật; fallback: patch thủ công rank sau build nếu rank_pattern không ăn | thấp |
| 9.12 | $\varepsilon$ ridge không có trong paper | $\varepsilon_{\text{rel}} = 10^{-6}$ (mục 3.4/7) | thấp |
| 9.13 | Cần thêm field config: `gora_gamma`, `gora_r_max`, `gora_use_rslora`, (tuỳ) `gora_paper_literal_init`, `gora_lr_ratio` — `config.py` là contract, không tự sửa | Lập danh sách field đề xuất cho điều phối duyệt rồi mới bổ sung | vừa (chặn implementation nếu chưa duyệt) |

**Cần escalated lên điều phối/user ngay:** mục 9.9 (CAO). Các mục 9.1, 9.7, 9.8, 9.10, 9.13 khuyến nghị xác nhận nhanh cùng lúc.

---

## 10. Ghi chú nguồn

- Toàn bộ công thức mục 3 lấy từ ar5iv render của arXiv 2502.12171 (fetch 2026-11-01): §3.1 (Eq. 1–4), §3.2 (Eq. 5–8), §3.3 (Eq. 9–10, Algorithm 1), Appendix A.1–A.2, B.3, B.4, Tables 9–12.
- Sự kiện orchestrator đã kiểm chứng lại và khớp paper: $N=64$ ✓; $r_{\min}=r_{\text{ref}}/2$ ✓; $I(W)=\mathrm{avg}(|W\odot G|)$ ✓; $B_0 = -(A^{\mathsf T}A)^{-1}A^{\mathsf T}G$ ✓ (quy ước paper); GoRA không đụng $W_0$ ✓. Bổ sung từ paper: $r_{\max} = 4r_{\text{ref}}$, bảng $\gamma$, lr ratio 16:1, scaling $\alpha/\sqrt r$.
