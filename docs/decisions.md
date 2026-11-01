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
| X1 | Oi/Bp | *(chờ spec-agent)* | PiSSA sửa trọng số gốc — xung đột với GoRA-init | Chờ spec |
| X3 | Oi+G | *(chờ spec-agent)* | Init gradient theo từng khối GraLoRA | Chờ spec |
| X4 | A+G | *(chờ spec-agent)* | Optimizer chiếu theo từng khối GraLoRA | Chờ spec |
