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

## Kết quả verify hộp đen (lớp 3)

*(điền sau khi verify-agent báo — V1/V2/V3, kèm danh sách DEFERRED sang Kaggle session đầu)*

## Kết quả review

*(điền sau khi reviewer báo — findings + trạng thái xử lý)*

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
