# Ngân sách — hiệu chỉnh bằng số đo thực sau Gate 0 (E0-SMOKE-L, T4, 2026-11-01)

Nguồn: `results/E0-SMOKE-L-qwen05b-lr2e-4-r32-s1.json` (50 bước, 20k→2k mẫu, fp16, sdpa).

| Hạng mục | Ước lượng (plan) | **Đo thực (Gate 0)** | Ghi chú |
|---|---|---|---|
| Setup một session (clone+pip+model tải) | — | ~15 phút | 1153s tổng − 268s train+eval |
| 50 bước (batch 8×accum 4, seq 512) | — | 268s | gồm cả eval 20 câu GSM8K bên trong |
| Throughput train (0.5B, T4, fp16) | — | ~3.0–5.4k tok/s | 3052 tok/s nếu tính cả eval; sẽ tách số sau run dài |
| peak mem (0.5B fp16 + L r=32, batch 8) | <14 GB | **4.2 GB** | dư dả — có thể tăng batch/seq cho 1.5B |
| 1 run sweep (0.5B, 20k mẫu, 1 epoch = 625 bước) | 30–45 phút | **~55–60 phút** | 625×5.4s ≈ 56 phút + eval 2 phút |
| GoRA probe (N=64 batch) | — | ~2–3 phút (dự phóng) | 64 lần fwd+bwd; đo thật ở stage2-O |
| Overhead A (chiếu r×r mỗi bước) | — | chưa đo | đo ở stage2-A |
| Overhead Bp (snapshot swap + reg) | — | chưa đo | đo ở stage2-Bp |
| 1 run chính (1.5B) | 2–3 giờ | ~2.5–3 giờ (dự phóng ×3 từ 0.5B) | cần peak-mem check trước |
| Quota còn lại | ≥25 giờ (user) | — | |

## Hệ quả cho lập kế hoạch

- Với ~1 giờ/run/0.5B và 2 GPU: **~40–44 run/ngày giới hạn thời gian**, thực tế quota là ràng buộc chính.
- Stage 1 (quét LR cho L, 5 run + 2 run seed-kép) ≈ 7 giờ ≈ 3.5 giờ wall trên 2 GPU.
- peak mem 4.2 GB cho 0.5B → 1.5B (~3×) ≈ 12.6 GB < 14.6 GB usable — vừa, nhưng bật `gradient_checkpointing` nếu Bp (+W₀ snapshot/reg) đẩy lên.

## Cắt giảm nếu thiếu quota (plan mục 8, thứ tự ưu tiên)

1. Bỏ 1.5B, chỉ 0.5B.
2. Giảm seed cuối 3 → 2.
3. Bỏ k=4 của GraLoRA.
4. Bỏ HumanEval, giữ GSM8K + GLUE.

Mọi cắt giảm phải ghi decisions.md.
