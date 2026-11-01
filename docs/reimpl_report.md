# Báo cáo reimpl (clean-room) — trạng thái sau Giai đoạn 0/reimpl

Cập nhật: 2026-11-01.

## Tổng quan

| Thành phần | Nguồn | Commit nhánh | Unit test (lớp 1+2) | Hộp đen (lớp 3) |
|---|---|---|---|---|
| `L` (LoRA) | PEFT `LoraConfig` | main | n/a (PEFT) | n/a |
| `G` (GraLoRA) | PEFT `GraloraConfig` (v0.19.0+) | main | n/a (PEFT) | V1: k=1 ≈ LoRA — xem tests/blackbox |
| `O` (GoRA alloc) | reimpl từ spec/gora.md | stage2/single-o `fda2ae1` | 9/9 | V2 |
| `Oi` (GoRA init) | reimpl từ spec/gora.md | stage2/single-o `fda2ae1` | (chung bộ trên) | V2 |
| `A` (AltLoRA) | reimpl từ spec/altlora.md | stage2/single-a `a4c0c13` | 9/9 (altlora_plus cài sẵn) | V3 |
| `B/Bp/Br` (BA-LoRA reg) | reimpl từ spec/balora.md | stage2/single-b `0097fee` | 11/11 | V3 |

Sau merge main: **29/29 unit test pass** (torch 2.13 CPU + peft 0.20.0 local; phiên bản ghim peft 0.21.2 kiểm trên Kaggle — Gate 0).

## Quy trình đã chạy

1. spec-agent ×3 đọc paper (arXiv 2502.12171 / 2505.12455 / 2408.04556) → `spec/*.md` kèm mục "Điểm chưa xác định".
2. Điều phối + user chốt mơ hồ (docs/decisions.md D-002: rsLoRA off, BA-LoRA defaults theo paper, B-first, LR grid, (m+n), rescale clamp, PEFT-native kaiming).
3. impl-agent ×3 viết `reimpl_*.py` + test trong worktree riêng, không đọc mã upstream.
4. Merge 3 nhánh vào main; chạy lại toàn bộ test.
5. verify-agent hộp đen + reviewer (kết quả ở dưới).

## Kết quả verify hộp đen (lớp 3) — verify-agent 2026-11-01

| Kiểm định | Kết quả | Số liệu |
|---|---|---|
| V1: GraLoRA k=1 suy biến về LoRA (peft upstream) | **MATCH** | ΔW + forward diff = 0.00e+00; scaling s khớp; A-init cùng phân phối kaiming |
| V2-B: importance Eq.(5) I=avg abs(W⊙G) so mã tác giả | **MATCH** | rel = 0.0 (cùng union_mean, features=int(in+out)) |
| V2-C: dạng ngân sách (m+n)·r (D-002.5) | **KHỚP** | budget 1408 = Σ(m+n)·r_ref; trainable khớp Σ(m+n)·r_i |
| V2-C/D: luật làm tròn rank | **MISMATCH** | tác giả dùng **floor** (12.74→12), spec dùng round-half-up (→13); lệch chỉ ở biên |
| V2-E: init B (hướng + scale) | **MATCH hướng** | cosine 1.000000; nhưng tác giả dùng **γ=1.0**, spec Table 9 γ=0.05@r=8 (B lớn hơn 20×) |
| V3-B: AltLoRA numeric 1:1 | **DEFERRED** | mã tác giả standalone chạy nhánh fallback (không phải Algo 1); so trên Kaggle với integration thật |
| V3-C: randomized SVD (BA-LoRA) | **MATCH** | rel vs exact SVD = 1.52e-07; k giá trị σ khớp từng giá trị |

File kiểm định: `tests/blackbox/` (test_gralora_k1.py, run_gora_blackbox.py, DEFERRED_KAGGLE.md — 4 mục deferred với thứ tự + env).

## Kết quả review (cavecrew-reviewer, diff 7cc106d..HEAD, 2019 dòng)

**3 🔴 — đã sửa + test bổ sung (tests/unit/test_pipeline_groups.py):**
1. `pipeline._adamw_groups`: B params bị loại KHỎI optimizer khi `lr_ratio_ba=1.0` (mọi run AdamW chỉ train lora_A) → group B giờ LUÔN được thêm; segment-matching (`{"lora_B","gralora_B"} & set(name.split("."))`) thay vì substring.
2. `gralora_B` không bao giờ khớp `startswith` (tên thật có prefix `base_model.model.`) → cùng fix trên.
3. `_pissa_snapshot` không sống qua resume (CR của Bp tắt im lặng sau TimeGuard-resume) → snapshot được pin ngay trong `build()` sau `get_peft_model` (PiSSA init là SVD deterministic → session mới tái tạo đúng A0/B0).

**3 🟡 — đã sửa:**
4. `balora` schedule `warmup_ratio=0` → ZeroDivisionError → guard `w > 0`.
5. GoRA probe đặt requires_grad trên toàn model (+1–3 GB transient) + không try/finally → freeze non-targets + try/finally restore.
6. `balora` student forward chạy dù cả 3 λ=0 (+50% step time) → early return `{}`.

**Đã kiểm và SẠCH:** X3 slice mapping (einsum đối chiếu peft layer), NOTE 9.11 (pattern key full-path match được), D-X1b, fp32-islands/ridge (không rò dtype), AltLoRA cache/parity, các D-002.1/2/5/6, D-X2.

Chạy lại toàn bộ: **32/32 pass** (29 cũ + 3 contract test mới).

## Các độ lệch triển khai so với spec/paper (chốt)

- Xem docs/decisions.md D-002 (10 mục) — gồm cả phát hiện mới:
- **D-X1b**: `GraloraConfig` không có `init_lora_weights` → `Bp+G` raise NotImplementedError; stage 5 quyết định custom PiSSA-for-GraLoRA hay ghi độ lệch.
- AltLoRA: counters là python int (không tensor scalar); cache preconditioner re-read sau khi ghi ngược (đúng ở mức eps fp16); T1 dùng ridge 1e-2 do bài toán synthetic khuếch đại 1/λ (test-only).
- BA-LoRA: key reg là "consistency"/"diversity"/"svdr"; student self-forward (+1 forward/bước so với kế hoạch dùng logits từ loss forward — đúng fallback spec cho phép); k NLU = 5 cứng.
- GoRA: `alpha_pattern` đi kèm `rank_pattern` giữ s = α/r_ref hằng theo layer (so sánh L vs L+O sạch trục scale); G+O: r_global = k·round(r_ref/k), phân phối per-layer chỉ làm log.

## Rủi ro còn mở

- `rank_pattern` key full-path của Qwen trên PEFT 0.21.2: chỉ test được toy trên 0.20.0 local — **phải smoke Kaggle (Gate 0)**.
- Tác động fp16 T4 thật: mọi đảo chiều/chiếu đã fp32-island nhưng chưa chạy GPU thật.
- svdvals backward NaN khi σ trùng tuyệt đối (measure-zero) — đã comment trong code.
- Black-box đầy đủ (V2/V3 nếu DEFERRED) dời sang Kaggle session đầu.
