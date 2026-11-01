# Spec: AltLoRA — Optimizer luân phiên cho LoRA (component code `A`)

Nguồn: paper *AltLoRA: Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections* (Yu, Wang, Chen, Xue — NeurIPS 2025, arXiv 2505.12455v2). Spec viết clean-room từ paper; không tham chiếu mã tác giả. Mọi trích dẫn công thức kèm số phương trình trong paper (v2).

---

## 1. Mục tiêu & phạm vi

- **Mục tiêu**: đóng gói thuật toán AltLoRA thành `torch.optim.Optimizer` thao tác trên các cặp tham số adapter (A, B), đăng ký vào hook `make_optimizer` của `registry.py` với code `A`.
- **Ý tưởng chính**: LoRA cập nhật joint A,B làm gradient tương đương không xấp xỉ tốt full gradient. AltLoRA cập nhật **luân phiên**: một optimizer step chỉ cập nhật MỘT trong hai ma trận, gradient được **precondition** để xấp xỉ full gradient bằng phép chiếu (Theorem 1), momentum bậc nhất được **realign** qua không gian thấp hạng (Theorem 2). Bộ nhớ optimizer đúng budget LoRA: $\mathcal{O}(kr+dr)$ (Table 1).
- **Trong phạm vi (bắt buộc)**: Algorithm 1 (`altlora`, momentum bậc nhất).
- **Tùy chọn (flag)**: Algorithm 2 (`altlora_plus`, thêm second-moment kiểu AdamW). Ghi rõ chi phí: state optimizer **gấp đôi** (thêm $V^A \in \mathbb{R}^{r\times d}$, $V^B \in \mathbb{R}^{k\times r}$ — vẫn $\mathcal{O}(kr+dr)$ nhưng 2 buffer/tham số thay vì 1). AltLoRA+ **mất** transformation invariance (paper mục 4.2 thừa nhận).
- **Ngoài phạm vi**: lý thuyết hội tụ, MoE nhiều expert, thực hiện phép chiếu lên full gradient kích thước $k\times d$.

## 2. Ký hiệu & shapes (đối chiếu PEFT)

Paper: $W_0 \in \mathbb{R}^{k\times d}$ (k = out, d = in), $\Delta W = s\,BA$ với $B\in\mathbb{R}^{k\times r}$, $A\in\mathbb{R}^{r\times d}$, $s = \alpha/r$.

| Paper | PEFT (v0.21.2 `lora.Linear`) | shape |
|---|---|---|
| $A$ | `lora_A.weight` | $(r, d_{in})$ |
| $B$ | `lora_B.weight` | $(d_{out}, r)$ |
| $s$ | `layer.scaling` = `lora_alpha / r` | scalar |

`AdapterContext.ab_pairs` chứa đúng cặp `(lora_A.weight, lora_B.weight)` cho mỗi module đã thích ứng — khớp 1:1 với ký hiệu paper.

**Preconditioner — kiểm tra chiều (quan trọng):**
- A-step: $B_t^\top B_t \in \mathbb{R}^{r\times r}$ (vì $B_t$ là $k\times r$), nhân **trái** gradient $A$ ($r\times d$). ✅ đúng như orchestrator dự đoán.
- B-step: $A_{t+1}A_{t+1}^\top \in \mathbb{R}^{r\times r}$ (vì $A$ là $r\times d$, nhân $A A^\top$ theo thứ tự đó), nhân **phải** gradient $B$ ($k\times r$).
- ⚠️ Sửa giả thuyết orchestrator: $AA^\top$ **không phải** $m\times m$; với layout PEFT/paper cả hai preconditioner đều $r\times r$. Cả hai đều rẻ: $\mathcal{O}((k+d)r^2)$ FLOPs, đúng time complexity $\mathcal{O}(kr^2+dr^2)$ của Table 1.

## 3. Công thức cập nhật

### 3.1 Gradient chuẩn từ autograd (Eq. 1, paper mục 2)

PEFT nhân $s$ trong forward nên autograd trả:

$$\nabla_A L = s\,B^\top(\nabla_W L), \qquad \nabla_B L = s\,(\nabla_W L)\,A^\top.$$

### 3.2 Preconditioned gradient (Theorem 1 + Corollary 1, Eq. 8/19)

Với ridge $\lambda$ (Corollary 1) — bản duy nhất dùng trong impl:

$$\tilde\nabla_{A_t}L = \frac{1}{s^2}\,(B_t^\top B_t + \lambda I_r)^{-1}\,\nabla_{A_t}L, \qquad
\tilde\nabla_{B_t}L = \frac{1}{s^2}\,\nabla_{B_t}L\,(A_{t+1}A_{t+1}^\top + \lambda I_r)^{-1}.$$

- Hệ số $1/s^2 = (r/\alpha)^2$ là **bắt buộc**: gradient autograd đã chứa $s$ (Eq. 1), Theorem 1 chia để về thang của $\nabla_W L$. Nếu dùng scaling khác chuẩn ($\alpha/r$, ví dụ rsLoRA $\alpha/\sqrt r$) công thức $1/s^2$ vẫn đúng vì $s$ chỉ xuất hiện qua forward — nhưng spec này chỉ chuẩn hóa cho $s=\alpha/r$.
- $\lambda = 10^{-6}$ theo paper (Appendix B.1.2: "1e−6 in our numerical studies, we don't tune the hyperparameter"). RIDGE LÀM MẤT TÍNH INVARIANCE CHÍNH XÁC với $Q$ không trực giao (xem §8.3).
- Khởi tạo $B=0$: ở B-step đầu, $A\sim$ Gaussian nên $A A^\top + \lambda I$ khả nghịch — OK. Ở A-step, $\nabla_A = sB^\top(\nabla_W L) = \mathbf{0}$ khi $B=\mathbf{0}$ ⇒ A-step là no-op bất kể preconditioner. **Hệ quả**: phải chạy B-step trước (xem §6.2).

### 3.3 Momentum bậc nhất + realignment (Theorem 2, Corollary 2, Algorithm 1)

Một optimizer step $t$ chỉ cập nhật một factor. Ký hiệu $B_{\text{old}},B_{\text{new}}$ = giá trị B trước/sau B-update gần nhất (tương tự $A_{\text{old}},A_{\text{new}}$).

**A-step** (update $A$; B giữ nguyên từ B-step trước):
$$\tilde M_t^A = (B_{\text{new}}^\top B_{\text{new}} + \lambda I)^{-1}\, B_{\text{new}}^\top B_{\text{old}}\, M_{t-1}^A$$
$$M_t^A \leftarrow \beta_1\,\tilde M_t^A + (1-\beta_1)\,\tilde\nabla_{A_t}L$$
$$A_{t+1} \leftarrow A_t - \eta\,(M_t^A + \gamma A_t)$$

**B-step** (update $B$; A đã được cập nhật ở A-step trước — đúng trạng thái $W_{t+1/2}=W_0+sB_tA_{t+1}$ của paper):
$$\tilde M_t^B = M_{t-1}^B\, A_{\text{old}} A_{\text{new}}^\top\,(A_{\text{new}} A_{\text{new}}^\top + \lambda I)^{-1}$$
$$M_t^B \leftarrow \beta_1\,\tilde M_t^B + (1-\beta_1)\,\tilde\nabla_{B_t}L$$
$$B_{t+1} \leftarrow B_t - \eta\,(M_t^B + \gamma B_t)$$

Ghi chú:
- Dòng $\tilde M_t^A$ trong Algorithm 1 viết $(B_t^\top B_t)^{-1}B_t^T B_{t-1}M_{t-1}^A$ với chỉ số lỏng lẻo; ngữ nghĩa: realign momentum từ không gian cột của $B$ **tại lần A-step trước** sang không gian cột của $B$ **hiện tại** — tức $B_{\text{old}}\to B_{\text{new}}$ như trên. Ridge cho phía A không được phát biểu tường minh (Corollary 2 chỉ nêu phía B) — áp dụng theo đối xứng (mức chưa xác định, §8.2).
- $\gamma$ là weight decay **gộp vào bước cập nhật** (dạng $-\eta\gamma A$): tương đương decay decoupled với hệ số $\eta\gamma$, KHÔNG đi qua mẫu số ở AltLoRA+. Mặc định thí nghiệm paper: $\gamma=0$.
- Thuật toán cập nhật **một factor / một optimizer step**; mỗi step một forward+backward trên batch mới, chỉ dùng gradient của factor đang hoạt động, bỏ gradient của factor kia (chi phí bỏ: r×d hoặc k×r phần tử — không đáng kể so với backward).

## 4. Giả mã

### Algorithm 1 — `altlora` (component A, chính)

```text
AltLoraOptimizer(pairs, lr=η, beta1=β1=0.9, weight_decay=γ=0.0, ridge=λ=1e-6,
                 start_phase="B", fp32_math=True)
# state per pair (A,B):
#   M_A ~ zeros(r,d) fp32,  M_B ~ zeros(k,r) fp32        (momentum)
#   P_A ~ (A Aᵀ + λI)⁻¹  cache r×r fp32   (tính tại A-step)
#   X_BA ~ A_old A_newᵀ  cache r×r fp32   (tính tại A-step, cho B-step sau)
#   P_B ~ (B Bᵀ + λI)⁻¹  cache r×r fp32   (tính tại B-step)
#   X_AB ~ B_newᵀ B_old  cache r×r fp32   (tính tại B-step, cho A-step sau)
# parity: bộ đếm nội bộ, chỉ tăng khi step() thực sự chạy

step():
  parity = counter % 2
  if parity == 0: phase = start_phase          # mặc định "B"
  else:          phase = start_phase.khác      # "A"

  for (A, B) in pairs:
    if phase == "B":
      g_B = B.grad (fp32)                       # đã unscale nếu fp16
      # preconditioner: dùng cache P_A, X_BA từ A-step trước (A không đổi từ đó)
      ∇̃_B = (1/s²) · g_B @ P_A⁻¹_cached
      if M_B != 0:  M̃_B = M_B @ X_BA_cached @ P_A⁻¹_cached
      else:         M̃_B = 0                     # lần đầu: momentum = 0, bỏ realign
      M_B = β1·M̃_B + (1−β1)·∇̃_B
      B_new = B − η·(M_B + γ·B)
      # cache cho A-step kế tiếp
      P_B⁻¹ = inv_fp32(B_newᵀB_new + λI);  X_AB = B_newᵀ @ B_old
    else:  # phase == "A"
      g_A = A.grad (fp32)
      ∇̃_A = (1/s²) · P_B⁻¹_cached @ g_A
      if M_A != 0:  M̃_A = P_B⁻¹_cached @ X_AB_cached @ M_A
      else:         M̃_A = 0
      M_A = β1·M̃_A + (1−β1)·∇̃_A
      A_new = A − η·(M_A + γ·A)
      P_A⁻¹ = inv_fp32(A_newA_newᵀ + λI);  X_BA = A_old @ A_newᵀ
  counter += 1
```

### Algorithm 2 — `altlora_plus` (flag `variant="altlora_plus"`)

Thêm state $V^A\in\mathbb{R}^{r\times d}$, $V^B\in\mathbb{R}^{k\times r}$ (fp32) và $\beta_2=0.999$, $\epsilon$ (mẫu số Adam):

$$V_t^A \leftarrow \beta_2 V_{t-1}^A + (1-\beta_2)(\tilde\nabla_{A_t}L \odot \tilde\nabla_{A_t}L),\qquad
A_{t+1} \leftarrow A_t - \eta\Big(M_t^A \oslash (\sqrt{V_t^A} + \epsilon) + \gamma A_t\Big)$$

(tương tự phía B). Theo đúng paper: **không bias correction**; mẫu số là $\sqrt{V}+\epsilon$, **không phải** $\sqrt{V+\epsilon^2}$ kiểu PyTorch Adam; decay thêm sau phép chia. Mất transformation invariance (paper mục 4.2 / D.2).

### Khởi tạo state

- $M_0^A = M_0^B = 0$; $V_0^A = V_0^B = 0$ (Algorithm 1/2 input). Hai lần dùng đầu của mỗi factor bỏ qua realignment (momentum bằng 0 ⇒ số hạng realign = 0; đồng thời tránh tham chiếu $B_{t-1}$ chưa tồn tại ở $t=0$ — paper không định nghĩa, xem §8.3).
- Cache $P_A, X_{BA}$ chưa tồn tại ở B-step đầu: vì $M_B = 0$ nên $\tilde M_B = 0$, nhưng $\tilde\nabla_B$ vẫn cần $P_A$ → tính trực tiếp từ $A$ hiện tại tại bước đó (không dùng cache lần đầu).

## 5. Siêu tham số mặc định

| Tham số | Giá trị mặc định | Nguồn / ghi chú |
|---|---|---|
| `altlora_momentum` $\beta_1$ | 0.9 | đã có trong `config.py`; paper dùng 0.9 |
| $\beta_2$ (chỉ altlora_plus) | 0.999 | paper mục 5.1 |
| ridge $\lambda$ | 1e-6 | paper Appendix B.1.2, không tune |
| $\epsilon$ (mẫu số altlora_plus) | 1e-6 (đề xuất) | paper KHÔNG nêu giá trị — xem §8.4 |
| weight decay $\gamma$ | 0.0 | paper dùng zero weight decay; khớp `config.weight_decay=0.0` |
| lr $\eta$ | **riêng cho AltLoRA** | xem dưới |
| `start_phase` | "B" | §6.2 |
| lr A vs B | **chung một lr** | Theorem 3: cùng $\eta=\mathcal{O}(1)$ cho A và B mới stable feature learning; thí nghiệm paper grid chung |

**lr logic**: AltLoRA (bậc nhất) trong paper đạt kết quả tốt nhất ở lr ∈ {1e-2, 1e-3, 1e-4} (grid mở rộng, Appendix E.1); AltLoRA+ dùng lr kiểu AdamW {1e-5, 4e-5, 1e-4}. `config.lr=2e-4` hiện tại là giá trị AdamW — với `["G","O","A"]` dùng altlora bậc nhất nên nâng lr (đề xuất 1e-3 làm điểm xuất phát). Vì A/B dùng chung lr, optimizer dùng **một param group** duy nhất; lr scheduler (cosine + warmup 0.03) ghi vào `param_groups[0]["lr"]` như optimizer thường — `step()` đọc `group["lr"]` mỗi lần gọi.

## 6. Điểm tích hợp

### 6.1 Hook `make_optimizer` (registry.py, code `A`)

- Module `src/lorastack/optim/altlora.py` tự đăng ký khi được import (theo cơ chế lazy của `registry.get_hooks`).
- `make_optimizer(cfg, model, ctx)`: bỏ qua `model`, thu thập `pairs = list(ctx.ab_pairs)` (đã được pipeline điền sau `get_peft_model`), trả về `AltLoraOptimizer(pairs, lr=cfg.lr, beta1=cfg.altlora_momentum, weight_decay=cfg.weight_decay, variant=cfg.altrlora_variant, ...)`. Cần thêm trường `altlora_variant: str = "altlora"` vào `ExpConfig` (không phá hợp đồng hiện có — thêm field mới với default).
- Pairing A↔B lưu trong optimizer bằng ánh xạ id(param) → pair index; `param_groups[0]["params"]` = [A₁,B₁,A₂,B₂,…] (thứ tự ổn định cho checkpoint).
- Class kế thừa `torch.optim.Optimizer` (để GradScaler/scheduler/checkpoint hoạt động chuẩn): implement `step(closure=None)` và để `zero_grad()` mặc định. `state_dict()`/`load_state_dict()` của base chỉ serialize `state`+`param_groups` → bộ đếm parity phải được lưu dưới dạng tensor trong `state` (gắn vào param đầu tiên) hoặc override state_dict; spec chọn: **lưu parity counter như tensor scalar trong state của cặp đầu tiên**.
- Factor không hoạt động trong step hiện tại: optimizer bỏ qua `p.grad is None`; nếu pipeline để cả hai factor require grad (mặc định PEFT) thì grad thừa bị bỏ qua và được dọn bởi `zero_grad(set_to_none=True)` bình thường. Phương án C.1 của paper (tắt `requires_grad` theo pha trong `training_step`) là tối ưu nhỏ về FLOPs — **không làm**, vì với grad_accum, tắt requires_grad phải đồng bộ với parity của optimizer step chứ không phải micro-batch (rủi ro lệch pha > lợi ích).

### 6.2 Trật tự pha đầu: B trước A

Algorithm 1 in trong paper bắt đầu bằng A (t mod 2 = 0), nhưng:
1. Bảng 5 (ablation): B-first tốt hơn A-first (GSM8K: 76.31 vs 74.49 cho AltLoRA); lý do paper nêu: với init chuẩn $B=0$, A-step đầu không có descent có nghĩa.
2. Với $B=0$: $\nabla_A = sB^\top(\nabla_W L)=\mathbf 0$ ⇒ A-step đầu là no-op tuyệt đối.

⇒ **Mặc định `start_phase="B"`** (step 0: B, step 1: A, …), cấu hình được. Khác biệt so với chữ Algorithm 1 được ghi ở đây một cách tường minh (mức chưa xác định §8.1).

### 6.3 GradScaler fp16 (T4)

- Huấn luyện fp16 chuẩn: `scaler.scale(loss).backward()` → `scaler.unscale_(optim)` → clip → `scaler.step(optim)` → `scaler.update()`.
- `scaler.unscale_(optim)` tự nhân 1/scale vào `p.grad` in-place — optimizer đọc `p.grad` như thường, **không cần hỗ trợ gì thêm**, MIỄN NI pipeline gọi `unscale_` trước `step` (HF Trainer tự làm; nếu pipeline tự viết loop thì phải gọi tường minh).
- `scaler.step` **không gọi** `optimizer.step()` khi phát hiện inf/nan ⇒ parity counter (nằm trong `step()`) không tăng ⇒ pha không lệch. Đây là lý do counter phải nằm trong optimizer chứ không phải trong pipeline.
- Trạng thái fp32: `M, V, P, X` luôn fp32 bất kể `param.dtype` fp16; cập nhật ghi xuống param bằng cast về dtype param.
- Gradient clipping (`max_grad_norm=1.0`) áp lên **grad thô** trước preconditioning (trong optimizer) — paper không đề cập; giữ nguyên hành vi pipeline (§8.6).

### 6.4 Tương tác với G (GraLoRA, xung đột X4)

⚠️ `docs/peft_inventory.md` **chưa tồn tại** trong repo → phần này là **giả định**, phải kiểm chứng khi impl GraLoRA lands:
- **Giả định**: GraLoRA chia weight thành `gralora_k` khối, mỗi khối là một cặp low-rank độc lập và `ctx.ab_pairs` sẽ chứa **một entry per khối** (kèm metadata khối nếu cần).
- **Thiết kế AltLoRA tương ứng**: preconditioning/momentum **theo từng khối độc lập** — mỗi cặp (Aᵢ,Bᵢ) có $P, X, M$ riêng, không có coupling chéo khối. Về mặt thuật toán này tự nhiên vì AltLoRA thao tác trên từng factorization riêng lẻ.
- **Rủi ro cần hỏi**: nếu GraLoRA chia sẻ A (hoặc B) giữa các khối, hoặc ΔW tổng là tổng các khối (mỗi khối có scale riêng), thì định nghĩa "full gradient" của một khối thay đổi và công thức $1/s^2$ cần hệ số scale của khối đó. → mở mục hỏi khi G sẵn sàng.

## 7. fp16/T4 — ổn định số học

- **Tất cả phép toán preconditioner fp32**: `Bf = B.detach().float()`; `P = Bf.T @ Bf + λ·I`; nghịch đảo qua **Cholesky** (`torch.linalg.cholesky` + `cholesky_inverse`, hoặc `cholesky_solve` cho vế phải) — KHÔNG dùng `eigvalsh`/SVD (đắt, dễ chết trên nửa phổ gần suy biến). $P$ đối xứng hóa `(P + P.T)/2` trước Cholesky.
- $\lambda = 10^{-6}$. Nếu Cholesky fail (LAPACK info > 0): nhân đệ quy $\lambda \leftarrow 10\lambda$ tối đa 3 lần rồi raise — hành vi hiếm gặp, ghi log. (Đề xuất; paper không nói fallback.)
- Kích thước $r\times r$ với r ≤ 128: nghịch đảo O(r³) ≪ chi phí backward (paper: O(r³) negligible, Table 3: 2h34m vs 2h13m của LoRA).
- Tránh nhân ma trận lớn không cần: $X_{AB} = B_{new}^\top B_{old}$ là r×r tính 1 lần tại B-step, tái dùng ở A-step sau.
- Cập nhật param: mọi biến tạm fp32, kết quả `.to(param.dtype)` rồi `param.data.add_(-1, update)` (hoặc `sub_`).

## 8. Điểm chưa xác định

| # | Điểm | Đề xuất | Ảnh hưởng |
|---|---|---|---|
| 8.1 | Paper Algorithm 1 in "A trước", nhưng ablation Table 5 + init $B=0$ chỉ ra B-first tốt hơn và A-first step 0 là no-op | `start_phase="B"` mặc định, config được | **Vừa** — hỏi user xác nhận trước khi chạy |
| 8.2 | Ridge cho realignment phía A (Corollary 2 chỉ phát biểu phía B); chỉ số $B_{t-1}$ trong $\tilde M_t^A$ lỏng lẻo | Áp ridge đối xứng; ngữ nghĩa "B tại lần A-step trước → B hiện tại" như §3.3 | Thấp |
| 8.3 | Ở $t=0$ paper không định nghĩa $B_{t-1}$ cho realignment (momentum = 0 nên vô hại về toán) | Bỏ qua realignment khi momentum = 0 | Thấp |
| 8.4 | AltLoRA+: giá trị $\epsilon$ không nêu; không bias correction | $\epsilon = 10^{-6}$ (an toàn fp16); theo đúng paper, không thêm correction | Vừa (nếu chạy altlora_plus) |
| 8.5 | lr mặc định `config.lr=2e-4` chưa hợp cho AltLoRA bậc nhất (paper dùng 1e-2…1e-4) | Thêm field `altlora_lr` hoặc set lr theo variant trong `make_optimizer`; đề xuất 1e-3 | **Vừa→Cao** cho kết quả thí nghiệm — hỏi user |
| 8.6 | Tương tác clip_grad (clipping grad thô trước preconditioner) và lr scheduler (paper dùng cosine, không chi tiết hóa) | Giữ pipeline hiện tại; ghi nhận sai khác | Thấp |
| 8.7 | GraLoRA X4: cấu trúc ab_pairs per-block là **giả định** (không có peft_inventory.md) | Per-block độc lập; verify khi G lands | Vừa |
| 8.8 | Ridge $\lambda>0$ phá invariance chính xác với $Q$ không trực giao: $(QAQ^\top+\lambda I)^{-1}\neq Q(AA^\top+\lambda I)^{-1}Q^{-1}$ | Chấp nhận sai số $\mathcal{O}(\lambda)$; test invariance dùng tolerance tương ứng | Thấp |
| 8.9 | Weight decay $\gamma$ gộp dạng $-\eta\gamma A$ (không đi qua mẫu số ở AltLoRA+) — đúng chữ thuật toán; hệ số hiệu dụng $\eta\gamma$ | Giữ $\gamma=0$ mặc định như paper | Thấp |

## 9. Tests tính chất (lớp 1, chạy CPU fp32)

File đề xuất: `tests/test_altlora_props.py`. Synthetic loss: least-squares $\mathcal{L}(A,B)=\frac12\|W_0 + sBA - Y\|_F^2$ với $k,d,r$ nhỏ (vd 16×24, r=4), gradient A/B lấy từ autograd (không hard-code công thức Eq. 1) để test cả hợp đồng gradient.

1. **Giảm loss đơn điệu** (β₁=0, η nhỏ): chạy T alternating steps; assert `loss[t+1] <= loss[t] + 1e-6` mọi t. Bất đẳng thức có thể vi phạm với momentum → test riêng β₁=0.
2. **Preconditioner đối xứng PSD sau ridge**: `P = BᵀB + λI` — assert `‖P − Pᵀ‖/‖P‖ < 1e-6`; Cholesky thành công; `P⁻¹P ≈ I` (sai số < 1e-4).
3. **Transformation invariance** (sai số O(λ), §8.8): $Q = I + 0.1R$ (well-conditioned, $Q$ khả nghịch r×r); pair 2 khởi tạo $(A', B') = (QA,\; BQ^{-1})$ sao cho $B'A' = BA$; chạy cùng chuỗi batch; assert `‖Δ(BA) − Δ(B'A')‖_F / ‖Δ(BA)‖_F < 1e-2` sau T bước. Chỉ có ý nghĩa khi cả hai pha đều chạy.
4. **Memory O(kr+dr)**: đếm numel toàn bộ `optimizer.state` (M, V nếu có, P, X) + tham số; assert tổng state **không chứa** tensor nào có numel ≥ k·d và tổng ≤ C·(kr+dr) với C nhỏ (vd ≤ 3 khi tính thêm O(r²) cache); riêng altlora_plus cho phép 2× phần M/V.
5. **Parity/pha**: sau 2·T step, A và B đều được cập nhật T lần; `scaler.step` bỏ qua (mô phỏng bằng grad=inf trước unscale) không làm lệch pha.
6. **fp32 math trên param fp16**: param dtype float16, assert state fp32 và không overflow (chạy 5 step với batch lớn gradient, không NaN).
7. **altlora_plus**: V cập nhật đúng công thức $\oslash(\sqrt V + \epsilon)$ (khớp 1 step tự tính tay), và KHÔNG khớp $\sqrt{V+\epsilon^2}$ — khóa hành vi theo paper.
