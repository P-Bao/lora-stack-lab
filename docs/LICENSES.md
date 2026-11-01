# LICENSES — nguồn gốc từng thành phần & khai báo minh bạch

Cập nhật: 2026-11-01.

## Thành phần của repo này

- `src/lorastack/`, `scripts/`, `configs/`, `notebooks/`, `tests/` (trừ `tests/blackbox/` phụ trợ): viết mới cho dự án — **Apache-2.0**.
- `spec/*.md`: đặc tả viết từ paper (nguồn của bản reimpl).

## Mã nguồn bên thứ ba

**Không có mã bên thứ ba nào được commit vào repo này.** `third_party/` nằm trong `.gitignore`, chỉ dùng để kiểm chứng hộp đen (`tests/blackbox/`, `@needs_upstream`) trong môi trường riêng.

### GraLoRA — có license (Apache-2.0)
- Upstream: https://github.com/SqueezeBits/GraLoRA — local snapshot commit `853e7c9b0d5c7eaa01767566ab53c14ef5479c5a` (2025-11-18).
- Tích hợp: qua PEFT (`GraloraConfig`, có từ PEFT v0.19.0, PR #2851). Không vendor.
- Lưu ý license chuỗi: repo gốc xây trên RaSA (MIT) + LLM-Adapters (Apache-2.0) — cả hai cho phép redistribute khi giữ notice.

### GoRA — CHƯA CÓ LICENSE (clean-room)
- Upstream: https://github.com/hhnqqq/MyTransformers — local snapshot commit `4037d4d6ba67ff88de87f90b943ff4e3a3649b67` (2026-03-25), không có file LICENSE.
- Xử lý: **không** đưa mã vào repo; viết lại từ paper (arXiv 2502.12171) theo quy trình spec → impl → verify hộp đen. Kết quả gắn nhãn `-reimpl`.

### AltLoRA — CHƯA CÓ LICENSE (clean-room)
- Upstream: https://github.com/LucasXinYu/AltLoRA — local snapshot commit `94b219882abcaa6004d2029fd8a09cd190a8a9aa` (2025-10-22); fork `peft/` bên trong là Apache-2.0 (giữ nguyên khi chạy hộp đen).
- Xử lý: như GoRA. Paper: arXiv 2505.12455.

### BA-LoRA — CHƯA CÓ LICENSE (clean-room)
- Upstream: https://github.com/llm172/BA-LoRA — local snapshot commit `1fe17ab3e39dcfefc630cdf386d69d942c61f16c` (2026-06-01); xây trên PiSSA (MIT).
- Xử lý: như GoRA; init PiSSA của `Bp` lấy từ API công khai PEFT (`init_lora_weights="pissa"`). Paper: arXiv 2408.04556.

## Khai báo minh bạch về phơi nhiễm mã (clean-room)

Trước khi thiết lập quy trình clean-room, **điều phối đã xem** (chỉ cấu trúc, không nghiên cứu logic thuật toán để tái sử dụng):
- Cây thư mục 4 repo upstream + README + file requirements của chúng;
- `AGENTS.md` của AltLoRA (mô tả cấu trúc file, bẫy phần cứng Kaggle — không chứa thuật toán);
- Lối vào các hàm chính (tên file: `optimizer_new.py`, `gora.py`, `train.py`) — chưa đọc thân hàm.

Bổ sung — **verify-agent (2026-11-01, hộp đen)** đã đọc hẹp để BIẾT CÁCH GỌI (không phải để hiểu thuật toán): grep dòng `def`/`class`; 4 range nhỏ trong `gora.py` (dòng 24–58, 298–315, 323–330, 370–383 — signature + call-site: `features=int(in+out)`, `grad_stored`=attribute, return 4-tuple); import-đầu-file 3 file trong `common/`; flags CLI của `example_llama3.1_metamathqa.sh` + defaults `gora_t5_helpers.py`. Ghi rõ để người đánh giá tự cân nhắc mức "clean".

Quy tắc áp dụng từ đó: `spec-agent` và `impl-agent` **không được đọc** bất kỳ file `.py/.sh/.ipynb` nào của 4 repo upstream; chỉ `verify-agent` chạy chúng như hộp đen trong môi trường tách biệt. Nếu quy trình này vẫn bị coi là phơi nhiễm không đủ sạch, phương án B (xin phép tác giả) là bước tiếp theo — đặt câu hỏi cho người dùng trước khi mở public.

## PEFT (Apache-2.0)

- Ghim phiên bản: xem `docs/decisions.md` D-001.
- Không fork, không vá gói đã cài; mọi hành vi mới đi qua API công khai hoặc được ghi vào `docs/decisions.md` như "độ lệch có chủ ý".
