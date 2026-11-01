# spec/balora.md — BA-LoRA: 3 regularizer (mã `B`, `Bp`, `Br`)

> Nguồn: paper *BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in LLMs*, ICLR 2026, arXiv 2408.04556 (đã fetch ar5iv, toàn văn gồm appendix A–D).
> Quy ước nguồn: `[paper §x / eq n]` = đọc trực tiếp từ paper. `[fact]` = [orchestrator-fact] — có trong contract/hướng dẫn dự án nhưng **không có** trong paper (khả năng từ code tác giả, chưa license → clean-room, chỉ theo mô tả).
> Cấm đọc code tác giả trong `{BA-LoRA,GoRA,AltLoRA,GraLoRA}/`. Mọi thứ dưới đây viết từ paper + fact.

---

## 1. Mục tiêu & phạm vi

- **Đích của method**: giảm "catastrophic inheritance" (nhiễu/mất cân bằng từ dữ liệu pre-training kế thừa vào fine-tune) bằng 3 regularizer trong **output space** (logits cuối của model), cộng vào task loss. PiSSA init chỉ là nền móng của BA-LoRA gốc `[paper §3.2]`.
- **Ba mã**:
  | Mã | Init | Regularizer | Ghi chú |
  |----|------|-------------|---------|
  | `Bp` | PiSSA (`init_mode="pissa"`) | 3 reg | = BA-LoRA đúng như paper (paper luôn dùng PiSSA `[paper §4.2]`) |
  | `Br` | mặc định LoRA, hoặc GoRA (`Oi`) nếu có | 3 reg | **độ lệch có chủ ý của dự án** (plan §1): paper không có biến thể này; tách ra để giải quyết X1 với GoRA init |
  | `B` | theo plan = đi kèm PiSSA | 3 reg | xử lý cùng `Bp` (registry map cả 3 mã → `losses/balora_reg.py`) |
- **Phạm vi task**: NLG (MetaMathQA→GSM8K, Magicoder→HumanEval) là biến thể chính; NLU (GLUE) có bộ công thức riêng `[paper §3.2.1]` — implement cả hai, dispatch theo `data.name`/`data.eval_name` (glue → NLU, còn lại → NLG).
- **Không thuộc phạm vi**: PiSSA init itself (dùng thẳng PEFT `init_lora_weights="pissa"`), phân bổ rank (`O`), optimizer (`A`).

---

## 2. Công thức 3 regularizer

Hàm mục tiêu tổng (cả NLU và NLG) `[paper eq 8, eq 12]`:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda_1 \mathcal{L}_{\text{CR}} + \lambda_2 \mathcal{L}_{\text{DR}} + \lambda_3 \mathcal{L}_{\text{SVDR}}$$

**Chiều tối ưu**: bài toán **min** tổng. Cả 3 reg là **penalty cộng với dấu `+`**:
- $\mathcal{L}_{\text{CR}} \ge 0$ — càng lớn càng lệch khỏi base → kéo về base.
- $\mathcal{L}_{\text{DR}}$ được định nghĩa **âm** của entropy (NLG) / bình phương hiệp phương sai (NLU) → min $\mathcal{L}$ ⇒ **max entropy / triệt tiêu tương quan**.
- $\mathcal{L}_{\text{SVDR}}$ là **số âm** của tỷ lệ khối lượng top-k singular values → min $\mathcal{L}$ ⇒ **max tỷ lệ đó**.

### 2.1 NLG (biến thể chính của dự án)

**Reg 1 — Consistency (distillation)** `[paper eq 9]`:

$$\mathcal{L}_{\text{CR\_NLG}} = \frac{1}{T}\sum_{t=1}^{T} \mathrm{KL}\big(\mathcal{P}_{\text{pt}}(y_t \mid y_{<t}, x)\ \|\ \mathcal{P}_{\text{ft}}(y_t \mid y_{<t}, x)\big)$$

- Là **KL, trên logits CUỐI (lm_head)** của model hiện tại ($\mathcal{P}_{\text{ft}}$) và model pre-trained ($\mathcal{P}_{\text{pt}}$), trung bình theo vị trí token $t \in [1, T]$. Không phải MSE (MSE là biến thể NLU), không phải hidden states.
- Chiều KL theo công thức viết: $\mathrm{KL}(P\|Q)=\sum_i P_i \log\frac{P_i}{Q_i}$ với $P = \mathcal{P}_{\text{pt}}$ (target, **detach**), $Q = \mathcal{P}_{\text{ft}}$ (student, có gradient) — chiều distillation chuẩn (forward KL).
- **Base model frozen nào**: paper định nghĩa $\mathbf{F}_P$ là logits "produced by the **pre-trained model**" `[paper §3.2.1]` = model gốc **$W_0$ đầy đủ**, KHÔNG phải $W_{res}$ của PiSSA (model $W_{res}$ là một mạng khác, logits khác $W_0$). Với `Bp` điều này bắt buộc một bản copy frozen của $W_0$ (xem §3, §6). Với `Br` không cần copy (xem §3.2).
- `[fact]` `distill_temp = 2.0`: **không có** trong eq 9. Đề xuất (chuẩn KD): $\mathcal{P} = \mathrm{softmax}(\text{logits}/\tau)$ cho cả hai phía, nhân cả term với $\tau^2$. Đã liệt kê §9.

**Reg 2 — Diversity (entropy)** `[paper eq 10]`:

$$\mathcal{L}_{\text{DR\_NLG}} = -\frac{1}{T}\sum_{t=1}^{T}\sum_{i=1}^{N} P_{\text{ft}}(x_i \mid h_t)\log P_{\text{ft}}(x_i \mid h_t) = -\bar H(P_{\text{ft}})$$

- Entropy của **phân phối dự đoán token của model hiện tại** tại mỗi vị trí (đầu vào softmax là hidden cuối $h_t$ — tức chính logits cuối; không cần hook tầng ẩn). Chỉ dùng output của model ft, **không dùng base**.
- Tối ưu: min $\mathcal{L}_{\text{DR}}$ ⇔ **max entropy** (chống representation collapse / lặp).
- `[fact]` `top_k_entropy = 20`: paper dùng **toàn bộ vocab** ($N$). Đề xuất: chỉ lấy 20 logit cao nhất mỗi vị trí, softmax lại trên 20 giá trị đó, entropy trên phân phối 20 chiều; giữ option `top_k=None` = đúng paper. Đã liệt kê §9.

**Reg 3 — SVD-based** `[paper eq 11]`:

$$\mathcal{L}_{\text{SVDR\_NLG}} = -\frac{\sum_{i=1}^{k}\tilde\sigma_i}{\sum_{j=1}^{D}\tilde\sigma_j}$$

- SVD trên **ma trận logits ft gộp cả batch**: $Z \in \mathbb{R}^{M \times V}$ với $M$ = tổng số vị trí (không pad) trong batch, $V$ = vocab size. Một SVD **mỗi bước train**, trên output model hiện tại.
- $\tilde\sigma_1 \ge \dots \ge \tilde\sigma_D$: singular values **xấp xỉ bằng randomized SVD** (paper viện dẫn randomized SVD của Pasand & Bashivan `[paper §3.2.3, App D.1]` để giảm chi phí); giữ **top-$k$** giá trị lớn nhất ở tử số.
- Tối ưu: min $\mathcal{L}$ ⇔ max tỷ lệ $\sum_{\text{top-}k}\sigma / \sum_{\text{all}}\sigma$ — dồn năng lượng vào $k$ pattern chính, triệt tiêu pattern nhiễu tần số cao.
- `[fact]` `svd_frob_norm = True`: **paper dùng mẫu số $\sum_{j=1}^D \tilde\sigma_j$** (xấp xỉ nuclear norm). Cờ frob đề xuất nghĩa: mẫu số thay bằng $\lVert Z\rVert_F$ (Frobenius của **toàn bộ** ma trận logits, tính trực tiếp không cần SVD phần đuôi). Hai mẫu số không bằng nhau; giá trị λ3 tune với mẫu số nào thì đi với mẫu số đó. Đã liệt kê §9.
- $k$: paper NLG dùng $k=10$ `[paper §4.2]` — trùng `balora_svd_k=10` trong config.

### 2.2 NLU (biến thể GLUE)

**Reg 1 — Consistency = MSE giữa logits đã chuẩn hóa L2** `[paper eq 4]`:

$$\mathcal{L}_{\text{CR\_NLU}} = \left\| \frac{\mathbf{F}_p}{\|\mathbf{F}_p\|_2} - \frac{\mathbf{F}_f}{\|\mathbf{F}_f\|_2} \right\|_2^2$$

- $\mathbf{F}_p, \mathbf{F}_f$: logits cuối $(B \times C)$ của model pre-trained / ft; chuẩn hóa L2 theo **toàn tensor** (Frobenius; đề xuất — paper không nói chi hơn).

**Reg 2 — Diversity = covariance loss (VicReg-style)** `[paper eq 5–6]`:

$$\mathcal{L}_{\text{DR\_NLU}} = \frac{1}{D}\sum_{i \ne j}\left[C(\mathbf{F}_f)\right]_{i,j}^2, \qquad C(\mathbf{F}_f) = \frac{1}{M-1}\sum_{i=1}^{M}(f_i - \bar f)(f_i - \bar f)^{\mathsf T}$$

- Hiệp phương sai **theo chiều batch** ($M$ = số mẫu, $D$ = số chiều output/logits); chỉ các phần tử **ngoài đường chéo**, bình phương, trung bình theo $D$. Không dùng entropy ở biến thể NLU.

**Reg 3 — SVD: giống NLG (eq 7) nhưng SVD chính xác** (ma trận $B\times C$ nhỏ, không cần randomized):

$$\mathcal{L}_{\text{SVDR\_NLU}} = -\frac{\sum_{i=1}^{k}\sigma_i}{\sum_{j=1}^{D}\sigma_j}, \qquad k=5 \text{ (NLU)}\ \text{[paper §4.2]}$$

---

## 3. Kiến trúc giả mã

### 3.1 Điểm thu thập dữ liệu — CHỈ logits cuối

Cả 3 reg (NLG lẫn NLU) chỉ cần **logits cuối** của (a) model đang train, (b) model tham chiếu. **Không cần forward hook tầng ẩn, không cần "toàn hidden"**. `ctx.hooks` (adapters/base.py) dùng làm kênh truyền `logits` từ forward của pipeline sang `loss_terms` (dict tự do) — tránh forward 2 lần cho model ft.

```python
# lorastack/losses/balora_reg.py — KHUNG, viết lại khi implement
def _setup(cfg, model, ctx):
    # gọi 1 lần sau get_peft_model
    if ctx.init_mode == "pissa":                      # Bp
        # cần model tham chiếu = W0 nguyên bản:
        # pipeline deepcopy model TRƯỚC get_peft_model, giữ ở ctx.hooks["_ref_model"]
        # (fp16, eval, requires_grad_(False), không phải trainable)
        ref_mode = "copy"
    else:                                             # Br (default / gora init)
        ref_mode = "disable_adapter"   # base layer chưa bị sửa → tắt adapter = W0 forward
    ctx.hooks["_balora_ref_mode"] = ref_mode

def loss_terms(cfg, model, ctx, batch):
    logits_ft = ctx.hooks.pop("__logits__", None)     # pipeline đã forward và cất sẵn
    if logits_ft is None:                             # fallback: tự forward (chấp nhận tốn 1 lần)
        logits_ft = model(**batch).logits
    T_total = ctx.hooks["_total_steps"]; t = ctx.hooks["_global_step"]

    with torch.no_grad():                             # NHÁNH THAM CHIẾU: detach toàn bộ
        if ctx.hooks["_balora_ref_mode"] == "copy":
            logits_ref = ref_model(**batch).logits    # copy W0 (Bp)
        else:
            with model.disable_adapter():
                logits_ref = model(**batch).logits    # (Br) = W0 forward, không tốn bộ nhớ weights

    lam1, lam2, lam3 = schedule(cfg, t, T_total)      # §4
    terms = {}
    if lam1 > 0: terms["balora_cr"]   = lam1 * L_cr_nlg(logits_ref, logits_ft, mask, cfg.balora_distill_temp)
    if lam2 > 0: terms["balora_dr"]   = lam2 * L_dr_nlg(logits_ft, mask, cfg.balora_top_k_entropy)
    if lam3 > 0: terms["balora_svdr"] = lam3 * L_svdr(logits_ft, mask, cfg.balora_svd_k, frob=True)
    return terms        # dict đã nhân λ; pipeline: total = task_loss + sum(terms.values())
```

Randomized SVD theo chunk (bắt buộc — không materialize ma trận $M\times V$ fp32 đầy đủ, T4 16 GB):

```python
def rand_svd_topk(Z_rows_iter, M, V, k, p=5, niter=1, chunk=1024):
    q = k + p; G = randn(V, q, fp32)                  # sinh 1 lần
    Y = zeros(M, q, fp32)                             # Y = Z @ G, cộng dồn theo chunk
    for z in chunks:  Y[rows] = z @ G
    for _ in range(niter):                            # tùy chọn, tăng accuracy
        Y2 = Z @ Y ; Y = Z.T @ Y2   # chunked tương tự
    Q = orth(Y)                                       # (M, q) nhỏ
    B = zeros(q, V, fp32)                             # B = Q.T @ Z, cộng dồn theo chunk
    for z in chunks:  B[:, cols] = Q.T @ z
    s = svdvals(B)                                    # xấp xỉ top-q singular values của Z, giảm dần
    return s
# L_svdr = -(s[:k].sum()) / (||Z||_F chunked  nếu frob=True else s.sum())
```

### 3.2 Model tham chiếu — bản chất và chi phí

| Đường | `Bp` (PiSSA) | `Br` (default/GoRA init) |
|-------|--------------|--------------------------|
| Base layer sau init | $W_{res} = W_0 - A_0B_0$ (PEFT sửa trọng số gốc) | $W_0$ nguyên vẹn (GoRA không đụng $W_0$ — plan X1) |
| `disable_adapter()` cho ra | logits của model $W_{res}$ — **KHÔNG** phải pre-trained model | logits của **$W_0$** — đúng pre-trained model |
| Kết luận | **phải giữ bản copy frozen $W_0$** (đúng định nghĩa paper "pre-trained model") | **không cần copy** — `disable_adapter()` miễn phí |
| Bộ nhớ thêm | +1.0 GB (0.5B fp16) / +3 GB (1.5B fp16) weights | 0 |

**Kiểm chứng từ paper**: "output logits produced by the pre-trained model" `[paper §3.2.1, eq 4]` — pre-trained model là $W_0$, không phải $W_{res}$. Đề xuất thay copy bằng `disable_adapter` cho `Bp` **bị bác bỏ** vì không tương đương. (Bằng chứng phụ: bảng chi phí paper App D.1 — BA-LoRA tốn +6.7 GB và +27% thời gian so với PiSSA trên 7B, tương thích với việc giữ thêm model tham chiếu.)

### 3.3 Vị trí cộng loss trong train loop

```
pipeline mỗi bước:
  outputs = peft_model(**batch)                    # 1 forward CÓ grad
  ctx.hooks["__logits__"] = outputs.logits
  task_loss = outputs.loss                         # chỉ trên response (paper §4.2)
  terms = loss_terms(cfg, model, ctx, batch)       # §3.1
  total = task_loss + sum(terms.values())
  scaler.scale(total).backward()                   # GradScaler fp16 như plan §2
```

---

## 4. Siêu tham số & schedule

### 4.1 Giá trị mặc định (config.py — contract) so với paper §4.2

| Tham số | config.py (dự án) `[fact]` | Paper NLG | Paper NLU |
|---------|---------------------------|-----------|-----------|
| λ1 (consistency) | 0.025 | 1e-5 | 1e-1 |
| λ2 (diversity) | 0.005 | 1e-1 | 1e-2 |
| λ3 (SVD) | 0.005 | 1e-1 | 1e-2 |
| k (svd_k) | 10 | 10 | 5 |
| distill_temp | 2.0 | — (không có) | — |
| top_k_entropy | 20 | — (full vocab N) | — |
| svd_frob_norm | True | — (tử: Σσ̃, mẫu: Σ mọi σ̃) | như trái |

Config.py là chuẩn dự án (dùng khi chạy); cột paper là tham chiếu tái lập. Chênh lệch lớn ở λ — đã liệt kê §9.

### 4.2 Schedule

**Không có schedule nào cho λ trong paper** (paper dùng λ cố định theo task; "cosine/linear" trong §4.2 là schedule của **learning rate**). Toàn bộ mục này là `[fact]` + đề xuất dự án.

Cần bổ sung 4 trường config (thay đổi contract, phải được orchestrator duyệt trước khi code):

```python
balora_lambda1_schedule: str = "cosine"      # "none" | "cosine"
balora_lambda_focus_schedule: str = "two_phase"  # áp cho λ2, λ3
balora_lambda_warmup_ratio: float = 0.2
balora_lambda_ramp_up_ratio: float = 0.05
```

Đề xuất công thức (đặt $w = $ warmup_ratio, $q = $ ramp_up_ratio, $u = t/T \in [0,1]$):

- **cosine cho λ1** (warmup tuyến tính rồi cosine decay về 0):

$$\lambda_1(u) = \lambda_1 \cdot \begin{cases} u/w & u \le w \\[4pt] \tfrac{1}{2}\big(1 + \cos\big(\pi \frac{u - w}{1 - w}\big)\big) & u > w \end{cases}$$

  Kiểm: $\lambda_1(0)=0$, $\lambda_1(w)=\lambda_1$, $\lambda_1(1)=0$.

- **two_phase cho λ2, λ3** (warmup tuyến tính, giữ giá trị, ramp-down cuối):

$$\lambda_{2,3}(u) = \lambda_{2,3} \cdot \begin{cases} u/w & u \le w \\[4pt] 1 & w < u \le 1-q \\[4pt] (1-u)/q & u > 1-q \end{cases}$$

  Kiểm: $\lambda(0)=0$, $\lambda(w)=\lambda$, $\lambda(1-q)=\lambda$, $\lambda(1)=0$.

⚠ Cả hai công thức là **đề xuất** (tên "two_phase"/"ramp_up" mơ hồ) — mức ảnh hưởng CAO, hỏi user (§9.1).

---

## 5. Điểm tích hợp

### 5.1 Hook `loss_terms` của registry.py

- `loss_terms(cfg, model, ctx, batch) -> dict[str, torch.Tensor]` — **trả các term ĐÃ nhân $\lambda_i(t)$** (schedule chạy trong module, đọc step từ `ctx.hooks`). Pipeline chỉ cộng `sum(terms.values())` vào task loss — giữ pipeline generic, không biết gì về λ.
- Tên key: `"balora_cr"`, `"balora_dr"`, `"balora_svdr"` (xuất hiện vào JSON loss curve để so ablation).
- Module `losses/balora_reg.py` **self-register cả 3 mã** `B`, `Bp`, `Br` khi import đầu tiên (registry `module_by_code` đã map sẵn). Hành vi phân biệt qua `ctx.init_mode` + `"Bp" in cfg.components`, không phải 3 hàm khác nhau.
- Nếu $\lambda_i(t) = 0$ (do schedule hoặc người dùng set 0): **skip luôn computation** của reg đó (tiết kiệm forward tham chiếu khi λ1=0).

### 5.2 Init PiSSA cho `Bp` qua PEFT v0.21.2

- `LoraConfig(init_lora_weights="pissa", ...)`: PEFT SVD từng trọng số $W$, đặt $A = U_{[:,:r]}S_{[:r,:r]}^{1/2}$, $B = S_{[:r,:r]}^{1/2}V_{[:,:r]}^{\mathsf T}$ (đã khớp scaling alpha/r bên trong), rồi **sửa base layer**: $W \leftarrow W - AB = W_{res}$. Đúng công thức eq (1)–(3) paper `[paper §3.1]`. `init_lora_weights="pissa_niter_N"` = fast SVD (vài giây, hơi sai số).
- Tương thích: chạy được với `rank_pattern` (cấp rank từng lớp của `O` — mỗi lớp SVD với rank riêng của nó). **Lưu ý PEFT**: chuyển đổi PiSSA→LoRA không hỗ trợ khi `use_rslora=True` + `rank_pattern` — dự án không dùng rslora → OK.
- **Lưu/load để resume** (ràng buộc plan §2: chỉ lưu adapter, không lưu model):
  1. Ngay sau `get_peft_model`: `peft_model.save_pretrained(ckpt/<ID>/pissa_init)` — adapter khởi tạo (nhỏ), giữ lại $A_0, B_0$.
  2. Checkpoint giữa chừng: `save_pretrained(...)` **không chuyển đổi** → lưu $A, B$ dạng PiSSA (hạng $r$).
  3. Resume: `PeftModel.from_pretrained(base_nguyên_bản, ckpt_dir)` — PEFT **tự chạy lại SVD** trên $W_0$ để dựng $W_{res}$ rồi nạp $A, B$ đã train. SVD đầy đủ ("pissa") là **xác định** → resume chính xác. Fast SVD (`pissa_niter_N`) dùng projection ngẫu nhiên → resume lệch nhẹ số học; nếu cần resume chuẩn xác tuyệt đối thì dùng "pissa" đầy đủ (0.5B: ~vài phút, chấp nhận được một lần).
  4. Sản phẩm cuối để chia sẻ: `save_pretrained(dir, path_initial_model_for_weight_conversion="ckpt/<ID>/pissa_init")` → adapter dạng LoRA thuần với $\Delta W = AB - A_0B_0 = [A \mid A_0][B \mid -B_0]^{\mathsf T}$ (hạng $2r$), nạp được lên base **nguyên bản không SVD**.
- `ctx.init_mode = "pissa"` do pipeline set; `pissa_init` KHÔNG chạy khi có `Oi` (X1).

### 5.3 X1 — Oi và PiSSA không bao giờ cùng bật

Validation ở pipeline (không phải trong balora_reg): `{"Oi"} ∩ {"Bp","B"} ≠ ∅` → raise ngay từ khi đọc config. `Br` + `Oi` hợp lệ (TETRAD-R).

---

## 6. Chi phí (kế hoạch đo)

Phân tích tĩnh — Qwen2.5-0.5B, $V = 151{,}936$, batch 8 × seq 512 ⇒ $M = 4096$:

| Hạng mục | `Bp` | `Br` |
|----------|------|------|
| Forward thêm / bước | 1 (no_grad, model tham chiếu) | 1 (no_grad, `disable_adapter()`) |
| Weights tham chiếu | **+1.0 GB** fp16 (copy $W_0$) | 0 |
| Logits tạm / bước | ~1.2 GB fp16 cho mỗi phía; chunk 1024 rows khi log-softmax/SVD ⇒ đỉnh ~0.6–1.2 GB | như trái |
| Randomized SVD | $2(1{+}niter)\cdot M V q \approx 40$–75 GFLOP fp32, $q=k{+}5=15$ ⇒ ~5–15 ms trên T4 (8 TFLOPS fp32) | như trái |
| KL + entropy + frob | vài ms (chunked) | như trái |
| Dự phóng thời gian/bước | **+30–45%** | **+25–40%** |
| Tham chiếu paper | 7B/A40: PiSSA 4h28 → BA-LoRA 5h41 (+27%), bộ nhớ +6.7 GB `[paper App D.1]` | — |

- **T4 16 GB**: 0.5B — training ~10 GB + 1 GB copy + ~1 GB tạm ⇒ vừa, cần theo dõi đỉnh. 1.5B — copy +3 GB ⇒ **bật `gradient_checkpointing`** khi chạy Bp.
- **Không chạy được ý "dùng output model gốc trước adapter thay vì copy" cho Bp** — base đã bị PiSSA sửa thành $W_{res}$; không có cách lấy logits $W_0$ từ model đã wrap (mạng phi tuyến). Chỉ `Br` được miễn (§3.2).
- **Đo thực tế** (bắt buộc, điền vào `docs/budget.md` ở Giai đoạn 5): `torch.cuda.max_memory_allocated()` và thời gian/bước, cặp run cùng config ± 3 reg, cho cả Bp và Br; so với dự phóng trên.

---

## 7. fp16 / T4 — ổn định số học

1. **SVD tính fp32**: cast chunk logits fp16→fp32 trước randomized SVD; `svdvals` trên ma trận $(q, V)$ nhỏ trong fp32. Không SVD trên fp16 (T4 không có lợi thế, dễ tràn).
2. **Entropy/KL trong log-space**: `log_softmax(logits_fp32)` rồi $-\sum \exp(\log p)\cdot \log p$; KL $= \sum p_{pt}(\log p_{pt} - \log p_{ft})$. Không bao giờ lấy `softmax` rồi `log` riêng. Chia `distill_temp` **trước** log_softmax.
3. **Detach đúng**: toàn bộ nhánh tham chiếu trong `torch.no_grad()`; `logits_ref.requires_grad == False` là bất biến; chỉ `logits_ft` giữ graph. Reg term cuối là scalar fp32 — an toàn với GradScaler.
4. **Chunk hóa mọi phép theo chiều $(M)$**: log_softmax, top-k, Frobenius, GEMM randomized SVD — đỉnh bộ nhớ tỷ lệ với chunk (mặc định 1024 rows ≈ 0.6 GB fp32).
5. Mask pad: mọi reg chỉ tính trên vị trí `attention_mask == 1`; mẫu số $T$/$M$ = số vị trí hợp lệ (tránh chia 0 khi batch toàn pad — guard `max(count,1)`).
6. NaN-guard: nếu $\lVert Z\rVert_F = 0$ hoặc $\sum\sigma = 0$ (batch suy biến) → trả 0 cho term đó (không chia 0).

---

## 8. Tests tính chất (lớp 1)

| # | Test | Kỳ vọng |
|---|------|----------|
| T1 | λ1=λ2=λ3=0, cùng seed/batch: `["L"]` vs `["L","B"]` | total loss **bằng nhau bitwise** (cộng 0·term); reg path không sinh NaN |
| T2 | CR suy biến: `logits_ft == logits_ref` | $\mathcal{L}_{CR} = 0$. Giá trị đã biết: $F_p=[1,0], F_f=[0,1]$ (đã normalize) ⇒ 2.0 |
| T3 | DR suy biến: phân phối đều trên top-k=20 | $\mathcal{L}_{DR} = -\log 20 \approx -2.9957$ |
| T4 | SVDR suy biến: ma trận có đúng $k$ singular values bằng nhau, còn lại 0 | mẫu số nuclear: $\mathcal{L}=-1$; mẫu số frob: $\mathcal{L}=-\sqrt{k}=-3.1623$ (k=10) — test ghi rõ cờ frob đang bật |
| T5 | Schedule: λ(u) tại $u=0, u=w, u=1-q, u=1$ | 0, λ, λ, 0 đúng theo công thức §4.2 (cả cosine lẫn two_phase) |
| T6 | `Br` không đụng $W_{res}$: hash state_dict base trước init và sau 1 bước train `["Br"]` | giống hệt nhau (hash bằng) |
| T7 | X1: config `["Oi","Bp"]` | raise validation error |
| T8 |Detach: backward total loss với Bp | không grad nào trên ref model; `logits_ref.requires_grad == False`; không graph leak giữa các bước |
| T9 | NLU dispatch: mini GLUE batch | CR_NLU theo eq 4 (MSE normalized), DR_NLU theo eq 5–6, k=5 |
| T10 | Resume Bp: save rồi `from_pretrained` lên base nguyên bản | logits khớp pre-save (full "pissa": exactly; niter: trong dung sai ~1e-3) |

---

## 9. Điểm chưa xác định (cần quyết / hỏi user)

| # | Vấn đề | Đề xuất của spec | Ảnh hưởng |
|---|--------|------------------|-----------|
| U1 | **Công thức schedule không có trong paper** (cosine λ1, two_phase focus, warmup 0.2, ramp 0.05 là `[fact]`, tên mơ hồ) | Chấp nhận công thức §4.2 (warmup→hold→ramp-down; warmup→cosine decay) | **CAO — hỏi user** |
| U2 | **`svd_frob_norm=True` mâu thuẫn paper** (paper: mẫu số = Σ mọi $\tilde\sigma$; frob: $\lVert Z\rVert_F$). λ3 tune với mẫu số nào thì phải dùng mẫu số đó | Bật frob theo `[fact]`, test T4 neo giá trị, ghi rõ trong kết quả | **CAO — hỏi user** |
| U3 | **λ mặc định dự án (0.025/0.005/0.005) khác paper (NLG 1e-5/1e-1/1e-1; NLU 1e-1/1e-2/1e-2)** — không rõ nguồn fact | Giữ config.py cho run dự án; ghi 2 bộ vào kết quả; coi λ là hyper cần sweep nhỏ ở Gate 2 | VỪA |
| U4 | `distill_temp=2.0` không trong eq 9; vị trí nhân $\tau^2$ | softmax(logit/τ) hai phía, nhân $\tau^2$ (chuẩn KD) | VỪA |
| U5 | `top_k_entropy=20` không trong eq 10 (paper: full vocab) | Top-20 renormalized + option `top_k=None` | VỪA |
| U6 | Ma trận SVD: toàn bộ vị trí không pad trong batch (1 ma trận $M\times V$) hay tách theo mẫu / chỉ response | Toàn bộ vị trí không pad — đọc nghĩa "batched output matrix" của paper | VỪA |
| U7 | Vị trí tính CR: tất cả vị trí hay chỉ response (paper chỉ nói task loss trên response) | Tất cả vị trí không pad, cùng mask cho 3 reg | THẤP |
| U8 | Chiều KL: paper viết KL(P_pt‖P_ft) — hiểu forward-KL (pt là target) | Theo đúng công thức viết | THẤP |
| U9 | Chuẩn hóa L2 trong eq 4: cả tensor (Frobenius) hay theo hàng | Cả tensor | THẤP |
| U10 | Số thành phần randomized SVD: oversample $p$, niter | $p=5$, niter=1; đo độ lệch vs SVD đầy đủ trong T4 | THẤP |

**Khó nhất (1 câu)**: tái dựng đúng "model pre-trained" cho consistency reg của `Bp` — paper ngầm định một bản copy $W_0$ frozen (không suy ra được từ model đã bị PiSSA sửa base), quyết định toàn bộ chi phí bộ nhớ và độ trung thực của biến thể `Bp` so với `Br`.
