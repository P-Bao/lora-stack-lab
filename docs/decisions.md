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
