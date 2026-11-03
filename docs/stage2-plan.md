# Giai đoạn 2 — Kế hoạch chi tiết (stage2 singles)

> Trạng thái: kế hoạch chốt 2026-11-03. Configs sinh sau khi user xác nhận grid.
> Nguồn: plan.md mục "Giai đoạn 2", decisions.md (D-002, D-003), findings.md (stage1).

## 0. Trạng thái vào

- Baseline **L\* = lr2e-4, r32, s1**: acc@500 triad = 45.0 / 45.0 / 43.5 → mean 44.5, **σ_L = 0.87**.
- **Gate 1 threshold = max(1.0, 2σ_L) = 1.73 điểm.** Đường tie = 42.8, đường improve = 46.2 (acc@500).
- Code reimpl đã merge trên main: `gora_alloc.py` / `gora_init.py` (O/Oi), AltLoRA (A), BA-LoRA (Bp/Br). 36 unit tests pass.
- Quota Kaggle: ~18.5h GPU còn lại trong tuần (reset 2026-11-02).

## 1. Điểm so sánh chuẩn (Bẫy 10)

- **Mọi Gate tính bằng acc@step-500** (reeval từ adapter snapshot — notebook tự chạy sau queue).
- Lý do: s1 stage1 chỉ có acc@500; acc@625 (s2 41.5, s3 42.5) thấp hơn ~2.5 điểm — nghi overfit nhẹ 125 bước cuối. So cùng bước tránh lệch hệ thống giữa stage1 và stage2.
- acc@625 direct vẫn ghi nhận làm tham chiếu phụ trong JSON.

## 2. Phương pháp & grid LR

| Method | Mã | Grid LR (r32, s1) | Runs | Căn cứ grid |
|---|---|---|---|---|
| GraLoRA (chỉ config, k=2) | G | {1e-4, 2e-4, 5e-4} | 3 | Cùng họ optimizer-scale với L → vùng quanh L\*=2e-4 |
| GoRA alloc + init | O | {5e-5, 1e-4, 2e-4} | 3 | Init B theo gradient → B lớn ban đầu → LR thấp hơn |
| GoRA init only | Oi | {1e-4, 2e-4} | 2 | Chung cơ chế init với O, tiết kiệm ngân sách |
| AltLoRA (B-first) | A | {2e-4, 1e-3, 5e-3, 1e-2} | 4 | D-002 #4: A bậc nhất cần LR cao |
| BA-LoRA PiSSA-init (W₀ frozen) | Bp | {1e-4, 2e-4, 5e-4} | 3 | Họ L + PiSSA init (paper PiSSA dùng vùng này) |
| BA-LoRA GoRA-init | Br | {1e-4, 2e-4} | 2 | Init-heavy như Oi |

**Tổng: 17 runs ≈ 17 GPU-h ≈ 8.5h wall (2 T4 song song).**

### Độ lệch có chủ ý so với plan.md (ghi khi báo cáo)

1. **3 LR thay vì 4** cho nhóm G/O/Oi/Bp/Br: stage1 chứng minh 5e-5 ≈ 1e-4 ≈ 2e-4 (lệch ≤3đ ≈ nhiễu) và 1e-3 diverge → grid 4 giá trị dư thừa. Chỉ A giữ 4 giá trị (D-002 #4).
2. **Bỏ single-b** (BA-LoRA init chuẩn): B không vào combo stage3 nào (TRIAD dùng Oi; TETRAD-P dùng Bp; TETRAD-R dùng Br). Nếu Bp VÀ Br cùng fail Gate 2 → bổ sung single-b (3 runs) vào giai đoạn điều tra để quy lỗi (reg hay init).
3. **Ablation G** (k=4; r=64 — plan.md yêu cầu k∈{2,4} × r∈{32,64}): 2 runs **điều kiện**, chỉ chạy nếu còn ≥3h quota sau 17 runs. So sánh cross-method luôn ở k2/r32.

## 3. Run IDs

```
stage2-G-lr1e-4-r32-s1   stage2-G-lr2e-4-r32-s1   stage2-G-lr5e-4-r32-s1
stage2-O-lr5e-5-r32-s1   stage2-O-lr1e-4-r32-s1   stage2-O-lr2e-4-r32-s1
stage2-Oi-lr1e-4-r32-s1  stage2-Oi-lr2e-4-r32-s1
stage2-A-lr2e-4-r32-s1   stage2-A-lr1e-3-r32-s1   stage2-A-lr5e-3-r32-s1   stage2-A-lr1e-2-r32-s1
stage2-Bp-lr1e-4-r32-s1  stage2-Bp-lr2e-4-r32-s1  stage2-Bp-lr5e-4-r32-s1
stage2-Br-lr1e-4-r32-s1  stage2-Br-lr2e-4-r32-s1
```

Config gốc: `configs/experiments/stage2-<CODE>-lr<X>-r32-s1.yaml`, override từ base.yaml (components, lr; Oi bật `gora_init_only`; Bp bật pissa + freeze W₀; A bật altlora B-first).

## 4. Session Kaggle — 2 phiên, dừng tay sau ô 3

### Phiên A — họ L (6 runs ≈ 3h wall)

1. Smoke 20 bước: G + Bp (~6 phút, bắt env bug trước khi đốt quota).
2. Thứ tự RUN_IDS (LR khả quan nhất trước — session chết giữa chừng vẫn có dữ liệu dùng được):
   `G-lr2e-4, G-lr1e-4, G-lr5e-4, Bp-lr2e-4, Bp-lr1e-4, Bp-lr5e-4`

### Phiên B — init-heavy + A (11 runs ≈ 5.5h wall)

1. Smoke 20 bước: O + Oi + Br + A (~12 phút).
2. Thứ tự RUN_IDS:
   `O-lr1e-4, O-lr5e-5, O-lr2e-4, Oi-lr1e-4, Oi-lr2e-4, Br-lr1e-4, Br-lr2e-4, A-lr2e-4, A-lr1e-3, A-lr5e-3, A-lr1e-2`

### Sau mỗi phiên

- Notebook tự reeval@500 (đã có sẵn từ commit 41d24ba).
- Tải `results_pack.zip` → đổ vào `wt-results\stage2\results\` (tạo mới) → commit nhánh results.
- **Kiểm tra `*.FAILED.json` ngay** — 1 run fail đã đốt ~1h quota.

## 5. Gate 2 (mỗi phương pháp — plan.md)

1. **(a) Cổng test 3 lớp**: reimpl đã qua unit test (36 pass). Re-run nếu đụng code.
2. **(b) Hiệu năng**: acc@500 của LR tốt nhất:
   - **≥ 46.2** → qua Gate 2, đánh dấu *improve* (Gate 1).
   - **42.8 – 46.2** → qua Gate 2 (tie), giữ làm ứng viên combo, ghi "không có lợi riêng lẻ".
   - **< 42.8** → KHÔNG qua: điều tra (reimpl sai? LR sai vùng? loss curve?) trước khi loại. Không đưa vào ghép.
3. Ghi kết luận từng method vào `docs/decisions.md` + `findings.md` (kèm loss step-620, acc@625).

## 6. Sau Gate 2 → Giai đoạn 3

- Chỉ ghép method qua Gate: `pair-go` (G+O, G+Oi — X2/X3), `pair-ga` (G+A — X4), `pair-oa` (Oi+A, O+A — X5), rồi TETRAD-P (G+O+A+Bp) / TETRAD-R (G+Oi+A+Br).
- Budget: sau stage2 còn ~0.5–1.5h quota tuần này → **stage3 chờ reset tuần sau** (30h).
- Stage3 cần quyết định trước: D-X1b (pissa-tự-viết cho GraLoRA hay init mặc định cho TETRAD-P).

## 7. Rủi ro & biện pháp

| Rủi ro | Biện pháp |
|---|---|
| 1 run FAILED đốt ~1h, margin mỏng | ≥2 run fail → dừng, điều tra, dời phần còn lại sang tuần sau |
| OOM Bp (copy frozen W₀, D-002 #9) | gradient_checkpointing đã bật sẵn (base.yaml); theo dõi VRAM run Bp đầu tiên |
| A-lr1e-2 diverge mạnh (NaN loss) | Bình thường — ghi nhận, KHÔNG rerun |
| Grid O quá thấp (5e-5 chậm) | Loss curve step-620 sẽ lộ; LR tốt nhất vẫn nổi trong {1e-4, 2e-4} |
| Session chết giữa chừng | run_queue skip done + thứ tự RUN_IDS ưu tiên LR khả quan nhất |
