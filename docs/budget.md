# Ngân sách — ƯỚC LƯỢNG THÔ, hiệu chỉnh bằng số đo thực sau Gate 0 (plan.md mục 0.7)

Điền cột "đo thực" ngay sau smoke test T4 (E0-SMOKE-L) + 1 run sweep đủ.

| Hạng mục | Ước lượng (plan) | Đo thực (Gate 0) | Ghi chú |
|---|---|---|---|
| 1 run sweep (0.5B, 20k mẫu, 1 epoch) | 30–45 phút | — | |
| 1 run chính (1.5B) | 2–3 giờ | — | |
| tokens/s (0.5B, T4, fp16, sdpa) | — | — | |
| peak mem (0.5B fp16 + adapter) | — | — | T4 usable ~14.7 GB |
| peak mem (1.5B fp16) | — | — | cần gradient_checkpointing? |
| Overhead O (N-batch probe) | — | — | N=64 |
| Overhead A (chiếu r×r mỗi bước) | — | — | O(r³) |
| Overhead Bp (snapshot swap + reg) | — | — | CR forward thêm |
| Tổng run toàn dự án | ~80 | — | plan mục 8 |
| Quota còn lại tuần này | ≥25 giờ (user) | — | |

## Cắt giảm nếu thiếu quota (plan mục 8, thứ tự ưu tiên)
1. Bỏ 1.5B, chỉ 0.5B.
2. Giảm seed cuối 3 → 2.
3. Bỏ k=4 của GraLoRA.
4. Bỏ HumanEval, giữ GSM8K + GLUE.

Mọi cắt giảm phải ghi decisions.md.
