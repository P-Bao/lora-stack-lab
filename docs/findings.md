# Kết quả & suy luận theo giai đoạn

> nhánh results — cập nhật sau mỗi stage, dùng cho báo cáo cuối
> Ngày: 2026-11-02

Dự án: đo xem chồng cải tiến LoRA (`GraLoRA + GoRA + AltLoRA + BA-LoRA`) có cộng hưởng không, đối chứng baseline `L` (PEFT LoRA gốc). Hạ tầng: Kaggle T4 x2, Qwen2.5-0.5B, MetaMathQA 20k train, GSM8K eval 200 mẫu.

---

## Stage 0 — Harness smoke

**Kết quả:** pass trên Kaggle (python 3.13.15, torch 2.11.0+cu128, 2×T4 14.6 GiB).

**Pin chốt (D-001):** `peft==0.21.2`, `transformers==4.57.6`, `accelerate==1.15.0`, `datasets==4.8.5`. torch KHÔNG ghim — dùng torch của image Kaggle.

**Bẫy phát hiện trong giai đoạn này:**

1. `torchao` 0.10.0 có sẵn trong image → `peft >= 0.19` raise dù không dùng torchao. Fix: xoá mọi `torchao*` khỏi `sys.path` trong bootstrap.
2. PyYAML đọc `2e-4` thành **str** → coerce trong `ExpConfig.from_dict`.
3. `default_collate` trả list → dùng collate tường minh, cố định độ dài.
4. `torch_dtype` deprecated → dùng `dtype`.
5. Session treo 12h, ba nguyên nhân chồng nhau:
   - `from_pretrained` không tự lên GPU → phải `model.to("cuda")`.
   - `DataLoader(num_workers=2)` fork sau khi HF/tokenizer mở thread → **deadlock** tại `next(data_iter)` đầu tiên, không log, TimeGuard không bắn → đặt `num_workers=0`.
   - TimeGuard check đưa **vào trong** vòng micro-batch; in tiến độ stdout mỗi 10 bước + `python -u`.
6. OOM 12.9 GB trên T4 14.6 (activations batch 8×512 + fp32 logits 2.32 GiB) → `gradient_checkpointing: true` (+35% step time) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
7. Loss in `0.000` do edit gỡ nhầm `running += loss.item()`; loss thật ở đầu train ~1.8x.

---

## Stage 1 — LR sweep cho baseline L

Cấu hình: r=32, seed 1, 5 LR `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}`, 625 bước, adapter snapshot bước 500.

### Lịch sử session

3 session đầu hỏng lần lượt: treo 12h → OOM → loss 0.000 (đúng các bẫy mục Stage 0). Session chạy thành công trên commit `dd8515d`, ~2.8h wall/run.

### Sự cố & rescue

- Run `lr5e-5` bị dừng tay **giữa eval** → mất json (còn `log.jsonl` + adapter).
- Cả 4 run có json đều báo `gsm8k_acc=0.0` → nghi bug hệ thống.
- **Rescue (2026-11-02):** KHÔNG train lại. Viết `scripts/reeval.py`: load base + `PeftModel.from_pretrained` từng adapter + `merge_and_unload` → eval lại → ghi `<ID>.reeval.json` + `<ID>.reeval.jsonl` (generation từng mẫu để chẩn đoán).
- Sửa gốc trong `evaluate()` (`evaluators.py`): `model.eval()` + `gradient_checkpointing_disable()` + batch generate 16 (padding_side="left").
- **Chẩn đoán xác nhận:** eval cũ chạy với gradient checkpointing còn bật → HF ép `past_key_values=None` trong generate (~100 phút/eval); `acc=0.0` là bug eval, **TRAINING KHÔNG LỖI**.

### Bảng kết quả reeval (GSM8K acc, n=200, adapter step-500)

| LR | GSM8K acc (%) |
|---|---|
| 5e-5 | 42.0 |
| 1e-4 | 42.0 |
| **2e-4** | **45.0 (best)** |
| 5e-4 | 39.5 |
| 1e-3 | 12.0 |

Wall reeval: ~2.3 phút/adapter (batch 16).

Loss train bước 620: `1e-4=0.484`, `2e-4=0.499`, `5e-4=0.593`, `1e-3=0.895`.

### Suy luận

- **(a)** `L*` = **lr2e-4** (45.0) — vừa cao nhất, vừa trùng default PEFT/paper.
- **(b)** 5e-5/1e-4/2e-4 lệch ≤ 3 điểm ≈ nhiễu binomial eval (σ ≈ 3.5 điểm @ n=200) — dải LR thấp tương đương.
- **(c)** 1e-3 diverge: loss 0.895 + acc 12.0 khớp nhau.
- **(d)** 5e-4 có dấu hiệu quá cao cho model 0.5B (acc thấp hơn dù loss tăng nhẹ).

### Hạn chế so sánh (Bẫy 10)

Acc bảng trên lấy từ **snapshot bước 500**. `evaluate()` cuối train (bước 625) hiện đã sửa → seeds 2/3 sẽ có acc bước-625 trực tiếp. Để so 3 seeds công bằng, **s2/s3 cũng phải reeval từ adapter bước-500**.

### Việc tiếp

Chạy seeds 2/3 của `L*` → tính `σ_L` → ngưỡng Gate 1 = `max(1.0 điểm, 2·σ_L)`.

---

## Stage 1 bổ sung — Seeds 2/3 & sigma_L (2026-11-03)

- Seeds 2/3 chạy trên commit `41d24ba` (notebook tự reeval bước-500 sau queue). Train ~60 phút/run, eval bước-625 trực tiếp đã sửa (batch 16).
- Kết quả L* (lr2e-4, r32):

| Seed | acc@500 (reeval) | acc@625 (direct) |
|---|---|---|
| s1 | 45.0 | — (eval cũ hỏng) |
| s2 | 45.0 | 41.5 |
| s3 | 43.5 | 42.5 |

- **sigma_L = 0.87** (n=3, sample stdev) → **Gate 1 threshold = max(1.0, 2·sigma_L) = 1.73 điểm**. Đường tie = 42.8, đường improve = 46.2 (đều tính trên acc@500).
- Quan sát: acc@625 thấp hơn acc@500 khoảng 2.5 điểm (s2 −3.5, s3 −1.0) — trong phạm vi nhiễu binomial (sigma ≈ 3.5 @ n=200), nghi overfit nhẹ 125 bước cuối. **Quyết định: mọi so sánh Gate dùng acc@500** (reeval từ adapter snapshot) để tránh lệch hệ thống giữa các stage.
- Baseline L* rất đồng nhất (sigma 0.87 << 3.5 nhiễu eval) → Gate 1 hẹp (1.73 điểm) là hợp lý, stage2 sẽ bị soi kỹ.
- **Stage 2 MỞ**: kế hoạch chi tiết tại `docs/stage2-plan.md` (nhánh main, repo lora-stack-lab).

---

## Stage 2 — Phiên A: singles G & Bp (2026-11-05)

Fix OOM Bp lần 2 trên commit `befefbc` (chunk kernel tự `index_select` hàng fp16 từ view logits, teacher probs tính lại trong chunk dưới `no_grad` — Bẫy 12). Notebook về bản tối giản RUN_IDS-only (`3e2861b`). Reeval chạy đủ cả hai phía: Bp ngay trong phiên train (json có sẵn → nhãn combo đúng), G qua dataset `stage2-adapter` (commit `527c2bc`: chỉ merge LoRA thuần, GRALORA eval nguyên trạng).

### Kết quả (GSM8K acc %, n=200, r32, s1, Qwen2.5-0.5B)

| Run | acc@625 (direct) | acc@500 (reeval) | peak_mem |
|---|---|---|---|
| G lr1e-4 | 42.5 | **50.0** | 9.5 GiB |
| G lr2e-4 | 45.5 | 47.0 | 9.5 GiB |
| G lr5e-4 | 35.0 | 41.0 | 9.5 GiB |
| Bp lr1e-4 | 41.0 | **37.5** | 11.3 GiB |
| Bp lr2e-4 | 29.0 | 33.5 | 11.3 GiB |
| Bp lr5e-4 | 16.0 | 18.0 | 11.3 GiB |

Smoke (n=20, chỉ xác nhận pipeline + memory profile): G 45.0, Bp 30.0.

### Gate 2 (improve ≥ 46.2, tie 42.8–46.2, fail < 42.8 — trên acc@500)

- **G: IMPROVE** — best 50.0 ≥ 46.2 → G vào mọi combo stage3. LR tốt nhất 1e-4, thấp hơn L* (2e-4) — xác nhận quét LR riêng (D-002); 5e-4 phá G (41.0).
- **Bp: KHÔNG QUA** — best 37.5 < 42.8 → điều tra trước khi loại (plan §5.3): Br (init GoRA thay vì PiSSA) chạy ở phiên B để quy lỗi reg-hay-init; Br cũng fail thì bổ sung single-b (3 runs) theo stage2-plan §2.2.

### Suy luận phụ

- Fix OOM xác nhận: peak 11578 MB (Bp) < 14.6 GiB, dư ~3 GiB; cao hơn ước tính lý thuyết 8–9 GiB nhưng an toàn.
- G: acc@500 (50.0) > acc@625 (42.5) — cùng chiều stage1 (overfit nhẹ cuối train), chốt chuẩn acc@500.
- Bp ngược chiều G (acc@500 < acc@625 ở lr1e-4/5e-4): nghi reg λ kéo mô hình về SVD basis sau warmup — chờ Br để quy lỗi.

---

## Stage 2 — Phiên B: A×4 xong, Oi/Br/O chết rồi fix (2026-11-05)

4 run A xong trên commit `7db7a54` (trước fix). Oi×2 + Br×2 chết ở build (`KeyError: 'Oi'` — Bẫy 13), O×3 chết ở probe backward (Bẫy 14) — fix trên `a627707` (nhánh main). Reeval + collect mất ở cả 3 phiên vì ô 2 `check=True` chết tại run_queue → json A tải tay từ `/kaggle/working`. Chạy lại mọi run còn lại (Br×2 + Oi×2 + O×3, ~10 GPU-h) trên commit `fa46280`.

### A — direct acc@625 (n=200, r32, s1, wall ~60–65'/run) — Gate 2 CHỜ reeval @500

| Run | acc@625 |
|---|---|
| A lr2e-4 | **48.5** |
| A lr5e-3 | 47.0 |
| A lr1e-2 | 47.0 |
| A lr1e-3 | 46.5 |

Suy luận tạm: A bền LR (spread 2.0 ≈ nhiễu binomial σ≈3.5) — grid 4 giá trị của A hóa ra dư nhưng rẻ. Best 48.5 @625 cao hơn cả G (45.5) và L* (44.5 @500) → A hứa hẹn; **chốt Gate 2 phải đợi reeval @500** (upload 4 adapter A thành dataset riêng, `STAGE="reeval"`, `COMPONENTS="A"`).

---

## Stage 2 — Phiên 7-run: Br/Oi/O xong 6/7, O5e-5 OOM (2026-11-05)

Chạy trên commit `fa46280` (sau fix Bẫy 13/14). 6/7 run xong; O-lr5e-5 chết OOM ở probe (Bẫy 15 — run_queue gán GPU theo parity, đè lên GPU đang có run train, 1 phút chồng lấp là đủ). Reeval trong phiên lại mất vì `check=True` (lần 4) → tải tay `/kaggle/working`.

### Br / Oi / O — direct acc@625 (n=200, r32, s1) — Gate 2 CHỜ reeval @500

| Run | acc@625 | wall |
|---|---|---|
| Br lr1e-4 | **48.5** | 114' |
| Br lr2e-4 | 43.5 | 114' |
| Oi lr1e-4 | **48.0** | 60' |
| Oi lr2e-4 | 45.0 | 60' |
| O lr1e-4 | 46.0 | 60' |
| O lr2e-4 | 42.0 | 60' |
| O lr5e-5 | OOM (Bẫy 15) | — |

Suy luận tạm: Br/Oi best ≈ A best (48.5), đều trên L* (44.5 @500) → 4 method còn lại đều hứa hẹn cho TRIAD/TETRAD-R. Br bền hơn Bp rõ (48.5 vs 37.5 cùng LR 1e-4) → GoRA init (Oi) khắc phục điểm yếu init PiSSA của Bp. O yếu nhất nhưng vẫn trên tie — O chỉ mang thông tin (không vào combo). O5e-5: hoãn rerun (O không vào combo; nếu cần đủ grid, nhét vào đầu phiên stage3).

**Kế tiếp:** reeval @500 cho 10 adapter (Br×2, Oi×2, O×2, A×4) qua dataset `stage2-adapter-all` (10 adapter + 10 run json; notebook đã set sẵn `STAGE="reeval"`) → aggregate + Gate 2 cho Br/Oi/O/A → stage3.

### Reeval @500 — Gate 2 CHỐT (2026-11-05)

Dataset mixed (16 adapter, slug `stage2-adapter`), 2 shard, ~25'. Nhãn components fallback "A" trên mọi file (Bẫy 16 — dataset thiếu run json) đã patch cục bộ; số đúng (eval deterministic — G/Bp trùng số cũ).

| Run | acc@500 | acc@625 | Gate 2 (improve ≥ 46.2, tie ≥ 42.8) |
|---|---|---|---|
| G lr1e-4 | **50.0** | 45.5 | **IMPROVE** |
| A lr5e-3 | **49.5** | 47.0 | **IMPROVE** |
| A lr1e-2 | 48.5 | 47.0 | IMPROVE |
| A lr2e-4 | 48.0 | 48.5 | IMPROVE |
| A lr1e-3 | 47.5 | 46.5 | IMPROVE |
| Br lr2e-4 | **47.5** | 43.5 | **IMPROVE** |
| Oi lr2e-4 | **47.0** | 45.0 | **IMPROVE** |
| Oi lr1e-4 | 45.5 | 48.0 | tie |
| O lr1e-4 | 46.0 | 46.0 | tie (46.0 < 46.2) |
| O lr2e-4 | 44.5 | 42.0 | tie |
| Br lr1e-4 | 44.0 | 48.5 | tie |
| Bp lr1e-4 (best) | 37.5 | 37.5 | FAIL |

Suy luận: (1) **cả 4 thành phần chồng được qua Gate 2** → TRIAD = G+Oi+A, TETRAD-R = G+Oi+A+Br; TETRAD-P chết; single-b không cần. (2) **A bền LR nhất** — 4/4 grid trên đường improve; G nhạy nhất (5e-4 → 41.0). (3) Br/Oi flip LR giữa @500/@625 (Br 1e-4: 48.5@625 nhưng 44.0@500) → nhiễu n=200 (σ≈3.5), không đọc quá kỹ thứ hạng LR; combo quét {1e-4, 2e-4} + 5e-4 edge. (4) O = tie đúng vai info-only (không vào combo). Aggregate sau khi bỏ E0-SMOKE (n=3 cho G/Bp): A 48.38±0.85, Oi 46.25, Oi+Br 45.75, O+L 45.25, G 46.00, Bp 29.67.

---

## Ghi chú quy trình

- Mọi kết quả JSON/log nằm trong nhánh `results` (thư mục `stage1/`, `stage2/`).
- Adapter (~336MB) không vào git — upload Kaggle Dataset `stage1-adapters`.
- Commit code trên nhánh `main` của `lora-stack-lab`.
