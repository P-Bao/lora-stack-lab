# Quyết định thiết kế (Decisions)

> Mỗi mục = 1 quyết định kèm bối cảnh + ngày kiểm chứng. Sửa quy ước nào phải ghi vào đây.

---

## D-001 — Chọn và ghim phiên bản peft / transformers / accelerate / datasets

**Ngày kiểm tra:** 2026-11-01 (PyPI JSON API + GitHub releases huggingface/peft).

### Phiên bản ghim

| Gói | Bản ghim | Lý do |
|---|---|---|
| `peft` | **0.21.2** | Bản ổn định mới nhất (release 01/10/2026). GraLoRA vào PEFT ở v0.19.0 (PR #2851); 0.21.2 có `GraloraConfig` đầy đủ (kèm `hybrid_r`) và đã qua 2 bản vá nhỏ (0.21.0 → 0.21.2, fix `target_parameters` và lỗi `add_adapter`/`load_adapter` nhiều adapter). |
| `transformers` | **4.57.6** | Bản 4.x ổn định mới nhất (PyPI, 2026-11-01). 5.x mới nhất là 5.22.1 — xem phân tích rủi ro dưới. |
| `accelerate` | **1.15.0** | Mới nhất; peft 0.21.2 yêu cầu `accelerate>=0.21.0` (setup.py). |
| `datasets` | **4.8.5** | 4.x ổn định mới nhất; tránh major 5.x (5.1.0) chưa kiểm chứng với luồng `load_dataset` của lab. |

### Lý do chọn transformers 4.x thay vì 5.x

- 4.57.6 là nhánh 4.x ổn định cuối; API Trainer / `get_peft_model` mà lab dùng đã bền bỉ qua nhiều bản.
- PEFT 0.21.2 **có** hỗ trợ transformers v5: `peft/import_utils.py` định nghĩa `is_transformers_ge_v5`, và `save_and_load.py` gọi `convert_peft_adapter_state_dict_for_transformers` khi load adapter dưới v5 — nhưng đây là đường chuyển đổi mới, **chưa smoke-test trên Kaggle**, nên Gate 0 không chọn.
- Mệnh đề "PEFT ≥0.18 cần cho transformers v5" được thoả (0.21.2 ≥ 0.18), tức nếu sau này nâng 5.x thì KHÔNG phải đổi peft.

### Fallback

- `peft==0.19.1` nếu 0.21.2 lỗi trên Kaggle (v0.19.1 là bản vá đầu của chuỗi đưa GraLoRA vào PEFT; đủ `GraloraConfig` cho mọi thí nghiệm G/O/G+O).
- Fallback chỉ đổi 1 dòng trong `requirements-kaggle.txt` + `tests/test_env.py` (`PEFT_PIN`).

### Lệnh cài

```bash
pip install -r requirements-kaggle.txt
python tests/test_env.py   # gate: exit 1 nếu lệch bản ghim
```

### Trạng thái

**Chưa smoke-test trên Kaggle — Gate 0 mới chốt cứng.** Mọi con số trên là kiểm tra qua web/source ngày 2026-11-01, chưa chạy trên máy thật T4 x2.

### Ghi chú torch

KHÔNG ghim torch: Kaggle preinstall ~2.x build đúng ABI NCCL/CUDA của image; `pip install torch` lại phá ABI (xem comment đầu `requirements-kaggle.txt`).

### Trạng thái sau Gate 0 (2026-11-01): ĐÃ SMOKE-TEST TRÊN KAGGLE — chốt cứng

- Kaggle image: python 3.13.15, torch 2.11.0+cu128, 2×T4 14.6 GiB — `peft==0.21.2` + `transformers==4.57.6` cài + chạy OK, `test_env.py` pass.
- **Bẫy 1 (đã ghi vào bootstrap):** image có sẵn `torchao` 0.10.0 ở `/usr/lib/python3/dist-packages`; peft ≥0.19 **raise** thay vì bỏ qua khi thấy torchao cũ — `pip uninstall` không gỡ được (outside-environment) → bootstrap quét xoá mọi `torchao*` trong mọi `sys.path` dir.
- **Bẫy 2:** `LoraConfig`/`get_peft_model` gọi dispatcher chuỗi có `is_torchao_available()` tính trước `isinstance` — xoá torchao là bắt buộc, không có cách bypass qua config.
- Bẫy 3 (PyYAML): `2e-4` đọc thành str — `ExpConfig.from_dict` ép kiểu theo annotation (fix 829af68).
- Bẫy 4: default_collate trả list cho batch → collate tường minh trong loaders.py (fix 7fa8872).
- Cảnh báo xung đột gradio/diffusers vs huggingface-hub 0.36.2: chỉ ảnh hưởng widget UI, bỏ qua.
- **Bẫy 6 (stage1 OOM 12.9 GB trên T4 14.6):** không bật `gradient_checkpointing` → activation batch 8×512 (~4–5 GB) + logits fp32 (2.32 GiB — chính là alloc bị từ chối) + trọng số fp16/grad (~2 GB) = ~13 GB. Sửa: `gradient_checkpointing: true` (base.yaml, +~35% step time) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (bootstrap).
- **Bẫy 7 (loss log `0.000`):** edit thêm `python -u` trong commit 6c98301 vô tình gỡ `running += loss.item()` khỏi vòng micro-batch → `running` luôn 0.0, mọi loss ghi/curve/in = 0.000 (KHÔNG phải loss thật nhỏ). Sửa 8f4f737. Chú ý: chạy test loss-curve, không chỉ 0.000.
- **Bẫy 5 (stage1 session chết 12h):** (a) `from_pretrained` không có `device_map` → model nằm trên CPU — phải `model.to("cuda")` tường minh; (b) `DataLoader(num_workers=2)` fork sau khi HF/tokenizer đã mở threads → **deadlock** tại `next(data_iter)` đầu tiên (không crash, không log, TimeGuard không bắn vì check nằm ngoài vòng micro-batch) → `num_workers=0` (fast tokenizer của Qwen tokenize 20k mẫu tính bằng giây); (c) TimeGuard check đưa VÀO trong vòng micro-batch; (d) train in stdout mỗi 10 bước + `python -u` trong run_queue để log Kaggle hiển thị tiến độ.
- **Bẫy 8 (notebook không tự dừng khi Save & Run All):** run cuối trong queue (lr5e-5) train xong nhưng bị dừng thủ công GIỮA eval → không có `<ID>.json` (chỉ còn log.jsonl + adapter step-500). Mọi stage phải chạy với đầu ra kiểm soát được; nếu dừng tay, kiểm tra JSON của run cuối trước khi tin kết quả.
- **Bẫy 9 (eval chậm ~10× do gradient checkpointing):** `pipeline.py` bật `gradient_checkpointing_enable()` cho train nhưng không tắt trước `evaluate()` → generate bị HF ép `past_key_values=None` (warning "Caching is incompatible...") → eval 200 mẫu × 512 tokens mất ~100 phút thay vì ~10. Cộng thêm `model.train()` không chuyển `model.eval()`. Sửa trong `evaluate()` (evaluators.py): `model.eval()` + `gradient_checkpointing_disable()` + batch generate (`eval_batch_size=16`, padding_side="left"). Thực tế stage1: wall_time 9976–10195 s/run, trong đó train ~54 phút, eval ~100 phút.
- **Bẫy 10 (không lưu adapter cuối train):** `train()` chỉ lưu adapter qua `save_ckpt` mỗi `ckpt_every_steps=500` bước — KHÔNG có save ở cuối training. Adapter trong ckpt/ là snapshot step 500/625. Chấp nhận cho stage1 (re-eval tất cả từ step-500 → so sánh công bằng); cân nhắc thêm save cuối train trước stage2 nếu cần đúng step 625.
- **Stage1 rescue (2026-11-02):** `gsm8k_acc=0.0` ở cả 4 run có JSON (n_eval=200) → nghi bug hệ thống phía eval/generation, KHÔNG tin các acc này. Cứu bằng `scripts/reeval.py`: load base + adapter (5/5 có sẵn, gồm lr5e-5) → eval lại → ghi `<ID>.reeval.json` + `<ID>.reeval.jsonl` (generation từng mẫu để chẩn đoán). `aggregate.py` ưu tiên bản reeval khi cùng ID có 2 bản. Đã chạy xong (commit 05a0f46): 5/5 adapter reeval thành công, ~2.3 phút/adapter (batch 16). Kết quả GSM8K acc (n=200, adapter bước-500): lr5e-5=42.0, lr1e-4=42.0, lr2e-4=45.0 (L*), lr5e-4=39.5, lr1e-3=12.0. Generation trong .reeval.jsonl ra CoT thật kết thúc `#### N` → acc=0.0 cũ xác nhận là bug eval (Bẫy 9), training không lỗi. Suy luận: 5e-5/1e-4/2e-4 lệch ≤3 điểm ≈ nhiễu binomial (σ≈3.5 điểm @ n=200); 1e-3 diverge khớp loss bước-620 (0.895). CHỐT: L* = lr2e-4 (45.0). Lưu ý: seeds 2/3 sẽ có acc bước-625 trực tiếp từ evaluate() đã sửa — để so 3 seeds công bằng phải reeval s2/s3 từ adapter bước-500.
- **Bẫy 11 (OOM khi chạy Bp, stage2 session A — 2026-11-04):** `torch.OutOfMemoryError` tại `_svdr_rand` L197, 13.44/14.56 GiB. Ba nguồn cộng hưởng: (a) `pipeline.train` forward `model(**batch)` cho task loss, rồi `balora_reg.loss_terms` **forward lại y nguyên batch đó** để lấy `logits_ft` → 2 graph student cùng sống tới backward (~1.2 GiB logits fp16 + activation mỗi cái); (b) các reg NLG (`_cr_nlg` KL, `_dr_nlg` entropy, `_svdr_rand` SVD) tạo trung gian **fp32 full-vocab** `(4096 hàng × 151643) ≈ 0.6 GiB/chunk` và autograd giữ tất cả chunk tới backward (~5-7 GiB retained) — chunking đơn thuần KHÔNG giảm retained; (c) fp32 log_softmax của loss chính (~2.3 GiB, Bẫy 6). Sửa: (1) pipeline truyền `logits_ft=out.logits` vào hook (bỏ forward thứ hai, math không đổi vì hai forward deterministic giống hệt); (2) `_ckpt()` bọc từng chunk bằng `torch.utils.checkpoint(..., use_reentrant=False)` → fp32 trung gian được tính lại lúc backward thay vì giữ; (3) `G` của randomized SVD tạo ngoài checkpoint (tránh RNG replay). Kỳ vọng peak ~13.4 → ~7-8 GiB. Test `test_t10` chặn regression: giá trị reg + gradient khi truyền logits phải bằng đường tự forward, và không được phát sinh forward thứ hai.
- **Bẫy 12 (Bp VẪN OOM sau fix Bẫy 11 — rerun session A, 2026-11-04):** 4/8 run fail, tất cả đều là Bp (smoke + 3 LR); OOM ở `backward` khi alloc đúng **594 MiB = 1 chunk fp32 `(1024 × 151936)`**. G (không reg) xong 625 bước, acc@625 0.455/0.425/0.35 — data G hợp lệ. Nguyên nhân: `torch.utils.checkpoint` **giữ MỌI tensor input tới backward**; fix Bẫy 11 truyền `lp_pt`/`p_pt` (teacher probs fp32 full-vocab, 2 × 0.58 GiB/chunk × 4 chunk ≈ **4.7 GiB**) làm input ⇒ chỉ đổi "giữ output autograd" lấy "giữ input", gần như không tiết kiệm (13.44 → 13.11 GiB). Sửa: mọi chunk kernel tự `index_select` hàng từ **view fp16 của logits** (`_flat` + `_mask_index`), teacher probs tính lại **bên trong** chunk (dưới `no_grad`) ⇒ input checkpoint chỉ còn tensor đã sống sẵn (logits, `ref`, index int64) + aux nhỏ (`G`, `Y`, `Q`) — bớt ~7 GiB retention. Thời điểm chết (~step 30) khớp lúc warmup λ kết thúc (reg bật). Test `test_t11` chặn regression: **không tensor fp32 cỡ vocab nào được làm input checkpoint**. Bài học: *checkpoint chỉ tiết kiệm khi input nhỏ*.
- **Bẫy 12b (log rác, cùng session):** `None of the inputs have requires_grad=True` (HF gradient checkpointing forward dưới `no_grad`: eval + ref forward của Bp/Br) và `` `use_cache=True` is incompatible with gradient checkpointing `` → xử lý tại nguồn: `filterwarnings` trong `lorastack/__init__.py` + `model.config.use_cache = False` lúc build (hành vi không đổi, chỉ sạch log). Notebook giữ **tối giản** (revert `3e2861b` sau hai lần thử thêm logic mount/audit): mỗi phiên Kaggle độc lập, chỉ cần sửa `RUN_IDS` cho đúng phần còn thiếu. Session A rerun: `E0-SMOKE-Bp` + 3× `stage2-Bp`; json/log/ckpt của G đã lưu `wt-results/stage2/` — acc@500 của G (Gate 2) cần một phiên `STAGE="reeval"` sau khi upload `ckpt/` thành dataset.
- **Stage1 hoàn tất (2026-11-03):** seeds 2/3 của L* xong (commit 41d24ba). acc@500 triad = 45.0/45.0/43.5 → **sigma_L = 0.87** → **Gate 1 threshold = max(1.0, 2·sigma_L) = 1.73 điểm** (tie ≥ 42.8, improve ≥ 46.2, tính trên acc@500). acc@625 thấp hơn ~2.5 điểm (nghi overfit nhẹ cuối train) → chuẩn so sánh mọi Gate = acc@step-500. Stage2 mở theo `docs/stage2-plan.md`.
- **Stage2 phiên A — Gate 2 cho G và Bp (2026-11-05):** cả 8 run xong trên commit `befefbc`. Fix OOM Bp xác nhận bằng số: `peak_mem_mb` = 11578 (Bp, mọi run) / 9730 (G) — trước fix chết ở 13.44 GiB + alloc 594 MiB fail. **G: IMPROVE** — acc@500 reeval = 50.0 (lr1e-4) / 47.0 (lr2e-4) / 41.0 (lr5e-4); best 50.0 ≥ 46.2 → G vào mọi combo stage3. **Bp: KHÔNG QUA** — 37.5 / 33.5 / 18.0; best 37.5 < 42.8 → chưa loại vội (plan §5.3), Br chạy phiên B để quy lỗi reg-hay-init; nếu Br cũng fail → bổ sung single-b (3 runs) theo stage2-plan §2.2. Ghi chú: LR tốt nhất của G là 1e-4 ≠ L* (2e-4) — xác nhận quyết định quét LR riêng (D-002); 5e-4 phá G. Reeval G chạy qua dataset `stage2-adapter` với `needs_merge` mới (commit `527c2bc`: chỉ merge LoRA thuần; GRALORA/pissa eval nguyên trạng). Chi tiết số: `wt-results/docs/findings.md` (nhánh results).
- **Bẫy 13 (module quên register — session B, 2026-11-05):** `gora_init.py` định nghĩa `init_adapter` nhưng **không có lệnh `register("Oi", ...)` nào trong file** → `get_hooks("Oi")` import module xong vẫn `KeyError: 'Oi'` → chết cả 2 run Oi VÀ cả 2 run Br (yaml Br là `components: [Oi, Br]` — chết ở bước init Oi trước khi kịp tới reg). Unit test pass vì test import thẳng hàm, không đi qua `get_hooks`. Sửa: `register("Oi", init_adapter=init_adapter)` cuối module + `test_t6_oi_self_registers_in_registry` (truy cập QUA `get_hooks`, không import trực tiếp).
- **Bẫy 14 (reentrant checkpoint cắt graph probe — cùng session):** probe GoRA chạy trên base model **chưa có adapter**, SAU khi `gradient_checkpointing_enable()` (pipeline build: bật ckpt ở bước 0, `allocate_ranks` ở bước 1). Reentrant checkpoint chỉ giữ graph khi **input** của block require grad — hidden states không require grad (embedding đóng băng; target weights là param, không phải input) → `loss` không có `grad_fn` → `backward` raise "element 0 of tensors does not require grad". L/G/Bp/A không chết vì lúc *train* có adapter params bên trong block. Sửa: `gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})` — non-reentrant cho grad chảy qua param; toán không đổi (các run đã xong không cần rerun). Test `test_t6b_probe_backward_survives_nonreentrant_checkpointing`.
- **Stage2 phiên B — A×4 xong, Oi/Br/O chết rồi fix (2026-11-05):** 4 run A xong trên commit `7db7a54` (trước fix): direct acc@625 = 48.5 (lr2e-4) / 47.0 (lr5e-3) / 47.0 (lr1e-2) / 46.5 (lr1e-3), wall ~60–65'/run — A bền LR (spread 2.0 ≈ nhiễu). **Gate 2 của A chờ reeval @500** (reeval + collect mất ở cả 3 phiên vì `check=True` chết tại run_queue — json A tải tay từ `/kaggle/working`). Oi/Br/O chết do Bẫy 13 + 14 → fix trên commit này, rerun B4a (Oi×2 + O×3) và B4b (Br×2).

---

## D-X2 — G+O: ngân sách rank khi GraloraConfig không hỗ trợ rank theo lớp

**Bằng chứng (source v0.21.2):**
- `tuners/gralora/config.py`: `GraloraConfig` KHÔNG có `rank_pattern` / `alpha_pattern` (chỉ `LoraConfig` có); chỉ có `layers_to_transform` / `layers_pattern` (lọc theo *chỉ số block*, không phải rank).
- `tuners/gralora/model.py` → `_create_and_replace`: `r = gralora_config.r` — r toàn cục, truyền thẳng vào `update_layer`, không tra pattern theo key.

**Quyết định:** khi ghép G+O:
1. Dùng **r toàn cục** (một `r` cho mọi `target_modules`) trong `GraloraConfig`.
2. O (GoRA alloc) vẫn chạy nhưng chỉ phân phối ngân sách **theo nhóm mức** (vd. nhóm attention / nhóm MLP, mức cao–trung–thấp); kết quả phân phối dùng để (a) chọn r toàn cục sao cho tổng tham số khớp ngân sách đã làm tròn theo `gralora_k`, (b) ghi log phục vụ phân tích.
3. Chênh lệch số tham số sau khi làm tròn/khung hoá được đo và ghi vào `AdapterContext.rank_budget_deviation` (trường đã có sẵn trong `src/lorastack/adapters/base.py`).

**Nhãn:** đây là **"độ lệch có chủ ý so với GoRA gốc"** (GoRA gốc phân rank theo từng lớp). Ghi rõ trong mọi báo cáo thí nghiệm G+O.

---

## Độ lệch có chủ ý (Intentional deviations)

Bảng trung tâm — spec-agent điền tiếp từ spec; không xóa dòng có sẵn.

| ID | Thành phần | Mô tả độ lệch | Lý do | Trạng thái |
|---|---|---|---|---|
| X2 | G+O | r toàn cục thay vì rank theo lớp; O chỉ phân phối ngân sách theo nhóm mức | `GraloraConfig` không có `rank_pattern` (bằng chứng: D-X2) | Đã quyết định |
| X1 | Oi/Bp | Oi và PiSSA loại trừ nhau (`pipeline.build` raise ValueError); PiSSA sửa W_res (xác minh `pissa_init`), GoRA-init không đụng W₀ | Hai cơ chế init không thể cùng bật | Đã quyết định |
| X3 | Oi+G | Init gradient gán THEO LÁT của param 3D `gralora_A [k, in//k, r]` / `gralora_B [k, r, out//k]` (không phải param riêng từng khối) | Cấu trúc thực của PEFT 0.21.2 (peft_inventory.md) | Đã quyết định |
| X4 | A+G | AltLoRA chiếu theo lát khối; PEFT `create_loraplus_optimizer` chỉ dùng làm baseline LoRA+ (group toàn tensor, bắt cả gralora_B) | peft_inventory.md | Đã quyết định |

## D-002 — Quyết định sau khi rà 3 spec (điều phối + user, 2026-11-01)

1. **rsLoRA (spec/gora.md 9.9 — user quyết định):** dùng **α/r cho MỌI config**, không bật `use_rslora`. `gora_use_rslora` (mặc định False) chỉ là flag ablation. Đây là độ lệch so với GoRA paper (dùng α/√r như một phần phương pháp) — ghi rõ khi báo cáo combo có Oi/O.
2. **BA-LoRA defaults (user quyết định):** theo PAPER — NLG: λ=(1e-5, 1e-1, 1e-1), k=10, SVDR mẫu số = Σσ̃ (bỏ frob_norm), entropy full-vocab (`balora_top_k_entropy=0`); NLU variant (k=5, MSE/cov/exact-SVD) chọn bằng `balora_variant`. Giá trị 0.025/0.005/0.005 trong README upstream KHÔNG dùng (repo chưa license + khác paper). Schedule không có trong paper → dùng đề xuất spec/balora.md §4.2 (`balora_schedule="spec42"`, warmup 0.2, ramp 0.05) — độ lệch có chủ ý.
3. **AltLoRA thứ tự pha (spec/altlora.md §8.1):** **B-first** (paper Table 5: 76.31 vs 74.49 GSM8K; A-first tại t=0 là no-op vì B=0). Field `altlora_order="B_first"`.
4. **LR quét cho single-a (spec/altlora.md §8.2):** AltLoRA bậc nhất cần lr cao hơn AdamW → stage2/single-a quét {2e-4, 1e-3, 5e-3, 1e-2} (quy tắc 0.3: mỗi phương pháp quét LR riêng).
5. **GoRA Eq.(7)–(8) (9.1):** hệ số dùng (m+n) theo bằng chứng nội tại của paper (mục 9.1 spec), không dùng √(m+n) của bản render.
6. **GoRA clamp rescale (9.7):** 1 vòng rescale ngân sách deterministic khi layer chạm r_min/r_max.
7. **GoRA hướng kaiming (9.8):** dùng PEFT-native init cho A (r×n), flag đổi hướng nếu cần đối chiếu.
8. **lr_ratio_ba (9.10):** field `lr_ratio_ba` (mặc định 1.0); GoRA App. B.3 dùng 16 — bật khi chạy combo có Oi theo paper, ghi rõ.
9. **Bp bộ nhớ (spec/balora.md §6):** consistency reg của Bp cần copy frozen W₀ (+1 GB/0.5B, +3 GB/1.5B) → bắt buộc gradient_checkpointing khi Bp trên 1.5B; Br miễn phí qua `disable_adapter()` (GoRA-init không đụng W₀).
10. **PiSSA resume (spec/balora.md §7):** resume qua `PeftModel.from_pretrained` trên base nguyên bản (PEFT tự chạy lại SVD); chuyển đổi PiSSA→LoRA (ΔW hạng 2r) chỉ dùng cho sản phẩm cuối.
11. **D-X1b — Bp + G (phát hiện từ impl-balora):** `GraloraConfig` KHÔNG có `init_lora_weights` → PEFT không cấp PiSSA cho GraLoRA. `pipeline.build` giờ raise NotImplementedError thay vì mất init âm thầm. Stage 5 (TETRAD-P = G+O+A+Bp) phải chọn: (a) viết `adapters/pissa_init.py` cho GraLoRA (SVD từng W, chia lát k×k — tái dùng logic lát của Oi) hoặc (b) TETRAD-P chạy init mặc định + ghi độ lệch. Quyết định trước khi mở Giai đoạn 5.

## D-003 — Quyết định từ verify hộp đen (user, 2026-11-01)

1. **Luật làm tròn rank (GoRA):** đổi round-half-up → **floor** (khớp hành vi quan sát được của mã tác giả qua hộp đen V2-C/D; lệch chỉ ở biên ≤1 rank/layer). 
2. **GoRA γ:** giữ **γ=0.05** mặc định (paper Table 9 @ r=8) — độ lệch có chủ ý so với mã tác giả thực chạy (γ=1.0 cho MetaMathQA, xác nhận qua hộp đen V2-E; init B nhỏ hơn 20×). Ghi rõ khi báo cáo combo Oi. Override bằng `cfg.gora_gamma`.
3. **AltLoRA numeric 1:1:** DEFERRED sang Kaggle session đầu (mã tác giả standalone chạy nhánh fallback, không tái hiện được Algo 1 local — xem tests/blackbox/DEFERRED_KAGGLE.md).
