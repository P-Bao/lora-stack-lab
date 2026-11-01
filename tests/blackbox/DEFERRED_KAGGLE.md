# Black-box verify (local pass) — DEFERRED list cho Kaggle session đầu

Ngày chạy local: 2026-11-01. Env local: Python 3.13.5, torch 2.13.0+cpu, peft 0.20.0 (upstream HF —
đÃ chứa GraloraConfig/GraloraModel), transformers 5.16.1. Không cài thêm gì, không sửa file tác giả.

## Đã MATCH/MISMATCH local (xem chi tiết trong reply của verify-agent)

| # | Kiểm định | Kết quả |
|---|-----------|---------|
| V1 | GraLoRA k=1 suy biến về LoRA (shapes transposed, dW, forward, scaling) | **MATCH** (max diff 0.00e+00, fp32 CPU) |
| V2-B | GoRA `compute_importance` (type='union_mean', scale=False, features=int(in+out), grad_stored=attr) vs Eq.(5) I=avg\|W⊙G\| | **MATCH rel=0.0** |
| V2-E | GoRA `dynamic_init` ΔW direction vs `_b0_solve`+ξ | **MATCH cosine=1.000000**; scale_fit c=20.0 = 1/γ_table → author γ=1.0, reimpl mặc định γ=0.05 (Table 9) |
| V2-C/D | `get_allocated_rank` budget = Σ(m+n)·r_ref (=1408, đúng dạng D-002.5) nhưng **floor** thay vì round-half-up | **MISMATCH 1 rank trên 1 layer ở biên (author 12, ours 13; raw=12.74; P-probes 12.5→12, 14.5→14)** |
| V3-C | BA-LoRA `BALoRATrainer._randomized_topk_singular_values` vs exact SVD (40×16 rank-5, k=5) | **MATCH rel=1.5e-07** |
| V3-B | AltLoRA `altlora` optimizer standalone | **DEFERRED** (fallback path, xem dưới) |

Lưu ý: strategy flag `gora_allocate_stretagy` (moderate/conserved/radical) KHÔNG đổi kết quả
trong path này → author thực chất luôn làm tròn xuống. Đây là lệch spec (spec 9.7 của mình:
round half-up) — cần user quyết định, KHÔNG sửa reimpl.

## DEFERRED cho Kaggle — thứ tự chạy đề nghị

Mọi script dưới đây ở `tests/blackbox/`, chạy bằng `python tests/blackbox/<tên>.py`, seed cố định,
không cần cài gì ngoài requirements-kaggle.txt.

1. **V2 bổ sung — GoRA `gora_reinit` end-to-end** (file: `run_gora_blackbox.py` hiện tại đã dừng ở
   cấp hàm; phần gora_reinit chưa có).
   - Vì sao DEFERRED local: cần CUDA (`torch.cuda.empty_cache()` trong gora_reinit), cần `paths.json`
     của MyTransformers (không được tạo/sửa local), cần deepspeed + bitsandbytes + immutabledict
     (đã stub local, trên Kaggle cài thật).
   - Env: T4/Tesla GPU, `pip install deepspeed bitsandbytes immutabledict pytz`, repo GoRA gốc nằm
     cùng cây thư mục, tạo `MyTransformers/paths.json` theo template repo tác giả (chỉ trên Kaggle).
   - Đo: 1 forward/backward probe → so named_importances + named_ranks của `gora_reinit` với
     `lorastack.adapters.gora_alloc.allocate_ranks` trên cùng batch; so `dynamic_init` B0 với
     `gora_init.init_adapter` (đã có cosine=1 local — kỳ vọng giữ nguyên).

2. **V3 bổ sung — AltLoRA numeric 1:1** (mở rộng `run_altlora_balora_blackbox.py`).
   - Vì sao DEFERRED local: `optimizer_new.altlora` chạy standalone nhưng rơi vào nhánh fallback
     ("no inverse", "no inversep2" — inverse của họ không dựng được trên harness tổng hợp), update
     CẢ HAI factor mỗi bước với delta hằng số ≈ plain-SGD, không tái hiện được Algorithm 1/2
     (alternating + preconditioner + momentum realignment). So sánh số từng bước cần path đúng.
   - Env: Kaggle T4, clone repo AltLoRA, chạy `main_lora.py` 1 step trên model nhỏ của HỌ
     (peft fork của họ qua sys.path, KHÔNG pip install đè), dump state (A, B, momentum) trước/sau
     step, rồi replay cùng (A, B, grad) vào `lorastack.optim.altlora.AltLoRAOptimizer` (beta1 khớp
     bằng cách đo từ I/O: betas=(0.999, 0.98) — xác định phần tử nào là β1).
   - Kỳ vọng đối chiếu: phase B đầu (B-first), hệ số (1−β1), hệ số 1/s², ridge λ=reg.

3. **V3 bổ sung — BA-LoRA `compute_loss` đầy đủ** (mở rộng `run_altlora_balora_blackbox.py`).
   - Vì sao DEFERRED local: `BALoRATrainer.compute_loss` cần Trainer + pt_model + tokenizer +
     batch HF thật; chỉ `_randomized_topk_singular_values` gọi được unbound (đã MATCH).
   - Env: Kaggle T4 + transformers + peft; model nhỏ (Qwen2.5-0.5B hoặc nhỏ hơn), batch synthetic
     4×64 tokens; bật từng lambda (λ1, λ2, λ3) riêng lẻ, so từng term với
     `lorastack.losses.balora_reg.loss_terms` (variant nlg), cùng mask/tau/k.

4. **GraLoRA fork của tác giả** (`GraLoRA/GraLoRA/peft`) — chưa đụng tới ở local pass vì peft
   upstream 0.20.0 đã ship GraloraConfig. Nếu user muốn so fork-vs-upstream: import fork qua
   sys.path trong subprocess (không cài đè), lặp lại V1 với `gralora_k>1` + `hybrid_r>0` và so
   forward/init với reimpl (unit tests hiện có của lab).

## Quy ước đã dùng ở local (tái lập trên Kaggle)

- Tác giả GoRA: `compute_importance(param, grad_stored, features, scale_features, type, lora_rank,
  max_lora_rank)` với `features = in_features + out_features` (int), `grad_stored` là ATTRIBUTE trên
  weight, `type` = 'union_mean', `scale_features` = False (mặc định gora_t5_helpers.py).
- `get_allocated_rank(model, args)` trả 4-tuple (total_budget, actual_trainable, named_ranks,
  named_importances); args cần: lora_rank, gora_max_rank, gora_min_rank, gora_importance_type,
  gora_scale_importance, gora_allocate_stretagy, gora_features_func, gora_softmax_importance.
- AltLoRA: `optimizer_new.altlora(model, lr, betas=(0.999,0.98), eps=1e-6, weight_decay,
  correct_bias, rank, reg)` — nhận model, tự gom cặp theo tên PEFT (lora_A.default.weight...).
- BA-LoRA: `train.BALoRATrainer._randomized_topk_singular_values(matrix, k)` — staticmethod,
  trả k singular values (không phải tỷ lệ khối lượng như reimpl của mình).
