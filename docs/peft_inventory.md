# Kho khảo sát PEFT v0.21.2 (clean-room, ghim cứng)

> **Phương pháp:** đọc trực tiếp source tag `v0.21.2` trên
> `raw.githubusercontent.com/huggingface/peft/v0.21.2/src/peft/...` ngày 2026-11-01.
> KHÔNG dựa trí nhớ, KHÔNG fork/patch gói peft. Số liệu trích kèm file gốc.

---

## 1. `LoraConfig` — `tuners/lora/config.py`

| Trường | Có | Kiểu / mặc định | Cách dùng / giới hạn |
|---|---|---|---|
| `r` | ✔ | `int = 8` | rank toàn cục; bị ghi đè bởi `rank_pattern` nếu khớp key |
| `lora_alpha` | ✔ | `int = 8` | scaling = `lora_alpha / r` (hoặc `/ sqrt(r)` nếu `use_rslora`) |
| `target_modules` | ✔ | `list[str] \| str \| None = None` | list = so khớp đuôi tên module; str = regex; `"all-linear"`; `[]` để chỉ dùng `target_parameters` |
| `exclude_modules` | ✔ | `list[str] \| str \| None = None` | loại trừ module đã khớp |
| `lora_dropout` | ✔ | `float = 0.0` | 0.0 → `nn.Identity()` |
| `rank_pattern` | ✔ | `dict = {}` | key so khớp regex ở **cuối** đường dẫn module (xem §4) |
| `alpha_pattern` | ✔ | `dict = {}` | như `rank_pattern` |
| `use_rslora` | ✔ | `bool = False` | True → scaling = `lora_alpha / sqrt(r)` |
| `use_dora` | ✔ | `bool = False` | DoRA; thêm `lora_magnitude_vector`; chi phí merge/forward lớn hơn |
| `init_lora_weights` | ✔ | xem dưới | giá trị hợp lệ (từ `update_layer` + Literal của field, v0.21.2): `True`, `False`, `"gaussian"`, `"eva"`, `"olora"`, `"pissa"`, `"pissa_niter_<số nguyên>"`, `"corda"`, `"loftq"`, `"orthogonal"` (yêu cầu r chẵn), `"mica"`, `"lora_ga"` (code chấp nhận khi có `LoraGAConfig`, dù không nằm trong Literal khai báo) |
| `layers_to_transform` | ✔ | `list[int] \| int \| None = None` | lọc theo chỉ số block |
| `layers_pattern` | ✔ | `list[str] \| str \| None = None` | dùng cùng `layers_to_transform` |
| `bias` | ✔ | `"none"` (mặc định) \| `"all"` \| `"lora_only"` | |
| `modules_to_save` | ✔ | `list[str] \| None = None` | bọc `ModulesToSaveWrapper` |
| `lora_bias` | ✔ | `bool = False` | bias trên LoRA B |
| `layer_replication` | ✔ | `list[tuple[int,int]] \| None` | xếp lớp lại, mỗi bản sao có adapter riêng |
| `runtime_config` | ✔ | `LoraRuntimeConfig(ephemeral_gpu_offload=False)` | không được save |
| `target_parameters` | ✔ | `list[str] \| None` | nhắm thẳng `nn.Parameter` (MoE); không dùng cho lab |
| khác | ✔ | `loftq_config`, `eva_config`, `corda_config`, `lora_ga_config`, `kasa_config`, `velora_config`, `monteclora_config`, `use_qalora`/`qalora_group_size`, `use_bdlora`, `arrow_config`, `alora_invocation_tokens`, `trainable_token_indices`, `megatron_config`/`megatron_core`, `ensure_weight_tying` | không dùng trong lab, liệt kê để biết có |

### PiSSA CÓ SỬA trọng số gốc (hệ quả X1)

`tuners/lora/layer.py` → `pissa_init()`: sau khi gán
`lora_A = diag(sqrt(Sr)) @ Uhr`, `lora_B = Vr @ diag(sqrt(Sr))` (Sr đã chia scaling), code làm:

```python
weight = weight.data - self.scaling[adapter] * lora_B @ lora_A
self.get_base_layer().weight.data = weight   # W_res thay W gốc
```

Hệ quả:

1. `base_model.state_dict()` **không còn là W gốc** — là W_res. `merge_and_unload()` cho lại W_full đúng (W_res + scaling·B·A).
2. `PeftModel.from_pretrained` với adapter PiSSA: `adapter_config.json` giữ `init_lora_weights="pissa"` → PEFT **chạy lại `pissa_init` trên base model mới** (tái tạo W_res và A₀,B₀) rồi mới nạp A,B đã train. Chỉ đúng khi r / lora_alpha / fan_in_fan_out khớp lúc train.
3. **X1:** PiSSA và GoRA-init (Oi) loại trừ nhau trên cùng model — Oi đo gradient trên W_res chứ không phải W gốc; ghi `init_mode` vào `AdapterContext` (đã có sẵn: `"default" | "gora" | "pissa"`).
4. `pissa_init` bắt dtype ∈ {fp32, fp16, bf16} (`TypeError` nếu khác) — OK cho T4 fp16. `"pissa_niter_<n>"` dùng `svd_lowrank` (nhanh, xấp xỉ).

---

## 2. `GraloraConfig` — `tuners/gralora/config.py`

| Trường | Kiểu / mặc định | Ghi chú |
|---|---|---|
| `r` | `int = 32` | rank **toàn cục** (xem §Kết luận) |
| `hybrid_r` | `int = 0` | >0 → thêm nhánh vanilla-LoRA riêng (`nn.Linear(in→hybrid_r→out)`); tổng tham số = `r + hybrid_r`; scaling = `alpha / (r + hybrid_r)`; =0 → `gralora_A_general/B_general` là `nn.Identity()` |
| `alpha` | `int = 64` | tên là `alpha`, **không phải** `lora_alpha` |
| `gralora_dropout` | `float = 0.0` | tên riêng, không phải `lora_dropout` |
| `gralora_k` | `int = 2` | số khối; khuyến nghị source: k=2 cho r≤32, k=4 cho r≥64 |
| `target_modules` | `list[str] \| str \| None = None` | như LoraConfig (hỗ trợ `"all-linear"`) |
| `fan_in_fan_out` | `bool = False` | tự chuyển True cho Conv1D (kèm warning) |
| `bias` | `"none"` \| `"all"` \| `"gralora_only"` | |
| `modules_to_save` | `list[str] \| None = None` | |
| `init_weights` | `bool = True` | tương đương `init_lora_weights` của LoRA nhưng **chỉ là bool**: True → A kaiming + B = 0 (no-op lúc init); False → cả A,B random (chỉ để debug/test) |
| `layers_to_transform` | `list[int] \| int \| None = None` | lọc theo chỉ số block |
| `layers_pattern` | `str \| None = None` | |
| (kế thừa PeftConfig) | `peft_type`, `task_type`, `inference_mode`, `base_model_name_or_path` | `task_type` hoạt động bình thường (§7) |

**KHÔNG có:** `rank_pattern`, `alpha_pattern`, `use_rslora`, `use_dora`, `init_lora_weights` (chuỗi), `exclude_modules`, `lora_bias`.

**Validate:**
- `config.__post_init__`: `gralora_k > 0`; `r % gralora_k == 0` (`ValueError`).
- `GraloraLayer.update_layer` (layer-level): `r > 0`; `hybrid_r >= 0`; **`in_features % gralora_k == 0`** và **`out_features % gralora_k == 0`**.

Với Qwen2.5-0.5B (hidden 896, intermediate 4864): chia hết cho 2 và 4 → r=32, k=2 hoặc k=4 đều an toàn.

---

## 3. CẤU TRÚC THAM SỐ BÊN TRONG GraLoRA — `tuners/gralora/layer.py` ⭐

Quyết định cho (a) init gradient theo từng khối (Oi với G, X3) và (b) optimizer chiếu theo khối (A với G, X4).

**Kết luận: mỗi khối là LÁT CẮT của MỘT tensor 3D duy nhất, KHÔNG phải param riêng.**

Cấu tạo trong `update_layer` (adapter_name mặc định `"default"`):

```python
# k khối được tạo riêng rồi torch.stack + transpose thành 1 param duy nhất
gralora_A = torch.stack([nn.Parameter(torch.empty(r, in_blk)) for _ in range(k)], dim=0)
gralora_A = gralora_A.transpose(1, 2).contiguous()   # [N, in//N, r]
gralora_B = torch.stack([nn.Parameter(torch.empty(out_blk, r)) for _ in range(k)], dim=0)
gralora_B = gralora_B.transpose(1, 2).contiguous()   # [N, r, out//N]
```

Tên attribute chính xác trên model (adapter `"default"`):

| Attribute | Kiểu | Shape tham số |
|---|---|---|
| `...q_proj.gralora_A.default` | `nn.ParameterDict` → 1 `nn.Parameter` 3D | `[gralora_k, in_features//k, r]` |
| `...q_proj.gralora_B.default` | `nn.ParameterDict` → 1 `nn.Parameter` 3D | `[gralora_k, r, out_features//k]` |
| `...q_proj.gralora_A_general.default.weight` | `nn.Linear(hybrid_r, in, bias=False)` khi `hybrid_r>0`, ngược lại `nn.Identity()` | `[hybrid_r, in]` |
| `...q_proj.gralora_B_general.default.weight` | `nn.Linear(hybrid_r, out, bias=False)` khi `hybrid_r>0` | `[out, hybrid_r]` |

Lưu ý 1: **khối A lưu dạng `[in_blk, r]` (đã transpose), khác quy ước LoRA chuẩn `[r, in]`**; khối B lưu `[r, out_blk]` (cũng khác LoRA chuẩn `[out, r]`).

Lưu ý 2: `self.module_name = module_name` — tên module đầy đủ được lưu trực tiếp trên layer, tiện cho log/hook.

**Cách chia khối (forward, v0.21.2):**
- Đầu vào được view thành `k` dải theo chiều feature: `x.view(B, L, N, in//N)`.
- Dải `n` chiếu bằng khối A `gralora_A[n]` lên **đầy đủ rank r** (`einsum "blni,nir->blnr"`).
- Rank r được reshape thành `[N, r//N]` rồi permute — đây là bước **trao đổi thông tin giữa các khối** (điểm khác GraLoRA vs block-diag LoRA).
- Chiếu xuống bằng khối B `gralora_B[n]` (`"bljr,jro->bljo"`), ghép lại thành đầu ra.
- Tổng tham số = `r*(in+out)` — đúng bằng LoRA cùng r; rank hiệu dụng tăng ~k lần.
- `merge`/`unmerge`/`get_delta_weight` có sẵn (scatter A + einsum) → `merge_and_unload()` dùng được; `supports_lora_conversion() → True`.

**Hệ quả thực thi cho lab:**
- **X3 (Oi với G — init gradient theo khối):** `.grad` của `gralora_A.default` là MỘT tensor 3D `[N, in_blk, r]`; grad của khối n = lát `gralora_A.default.grad[n]`. Không set `requires_grad`/optimizer riêng từng khối được vì chỉ có 1 param — mọi thao tác theo khối đều là **lát cắt index** `param[n]`.
- **X4 (A với G — optimizer chiếu theo khối):** `torch.optim` chỉ thấy 1 param 3D. Muốn state/chiếu theo từng khối: tự viết optimizer (registry `A`, module reimpl) lưu state shape `[N, in_blk, r]` / `[N, r, out_blk]` và xử lý từng lát trong `step()`; hoặc tách qua `param[n].contiguous()` + copy-back (tốn bộ nhớ, tránh). KHÔNG fork peft — bọc ngoài là đủ.
- fp16: forward cast `x` về dtype của `gralora_A`; `get_delta_weight` tự lên fp32 khi merge trên CPU fp16 — merge trên GPU T4 giữ nguyên fp16.

---

## 4. `LoraModel` — đường dẫn param — `tuners/lora/layer.py`

```python
self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
```

- Đường dẫn: `...q_proj.lora_A.default.weight` `[r, in]`; `...q_proj.lora_B.default.weight` `[out, r]` (nn.Linear nên có `.weight`).
- Embedding: `...lora_embedding_A.default` / `...lora_embedding_B.default` (`nn.ParameterDict`, 2D).
- `rank_pattern`/`alpha_pattern` resolution — `tuners/lora/model.py` `_create_and_replace`:
  ```python
  r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
  r = lora_config.rank_pattern.get(r_key, lora_config.r)
  ```
  với `get_pattern_key` (`peft/utils/other.py`): `re.match(rf"(.*\.)?({key})$", current_key)` — **pattern key là regex so khớp ở CUỐI đường dẫn module** (vd. key `"q_proj"` khớp mọi `...q_proj`; key có số thứ tự dùng nguyên văn, không tự thay `*`).

---

## 5. save_pretrained / load_adapter / PeftModel.from_pretrained

Áp dụng cho **cả** `LoraConfig` và `GraloraConfig` (đường generic):

- **Save:** `PeftModel.save_pretrained` → `get_peft_model_state_dict` (`utils/save_and_load.py`) → `tuner_cls._get_adapter_state_dict` (generic trong `BaseTuner`, dựa trên `adapter_layer_names` + liệt kê con của layer) → `adapter_model.safetensors` + `adapter_config.json`. Key bỏ tên adapter (`gralora_A.default` → `gralora_A.weight`; riêng GraLoRA key 3D không có `.weight` cho A/B khối: `gralora_A.weight` — xác nhận khi smoke-test Gate 0).
- **Load:** `PeftConfig.from_pretrained` → `PEFT_TYPE_TO_CONFIG_MAPPING["gralora"]` (đăng ký qua `register_peft_method` trong `tuners/gralora/__init__.py`) → inject adapter → `set_peft_model_state_dict` → `BaseTuner._remap_adapter_state_dict_for_load`. **Resume OK cho cả hai config.**
- **Merge:** `model.merge_and_unload()` — cả `LoraModel` lẫn `GraloraModel` kế thừa `BaseTuner._unload_and_optionally_merge`; `GraloraLayer.merge/get_delta_weight/unmerge` implement đầy đủ.
- PiSSA: see §1 — load chạy lại `pissa_init` trước khi nạp weights.
- Lưu ý v5: nếu sau này dùng transformers 5.x, `set_peft_model_state_dict` sẽ tự gọi `convert_peft_adapter_state_dict_for_transformers` — với 4.57.6 nhánh này không chạy (`is_transformers_ge_v5 = False`).

---

## 6. PEFT có sẵn optimizer kiểu LoRA+ không? — **CÓ** (khác kỳ vọng)

`src/peft/optimizers/` tồn tại ở v0.21.2 với 3 hàm (export từ `peft.optimizers`, **không** nằm trong `peft.__all__` gốc):

```python
from peft.optimizers import create_loraplus_optimizer  # + create_lorafa_optimizer, create_riemannian_optimizer
```

`create_loraplus_optimizer(model, optimizer_cls, *, lr, loraplus_lr_ratio, **kwargs)`:

- Nhóm tham số theo chuỗi: `"lora_B" in name` (so **substring**!) → groupB lr = `lr * loraplus_lr_ratio`; còn lại groupA lr = `lr`; group embedding lr riêng (`loraplus_lr_embedding`, mặc định 1e-6).
- **Cạm bẫy cho lab:** substring `"lora_B"` cũng khớp `"gralora_B"` — nhưng ở mức **toàn tensor 3D**, không theo từng khối; không dùng được cho X4. Component A (AltLoRA) tự viết optimizer qua registry, không dựa vào hàm này — chỉ dùng nó (nếu cần) cho baseline L với `loraplus_lr_ratio` như một biến đối chứng.

---

## 7. `get_peft_model` + `TaskType.CAUSAL_LM`

- `GraloraConfig` kế thừa `task_type` từ `PeftConfig` → hỗ trợ đầy đủ.
- `get_peft_model(model, GraloraConfig(task_type=TaskType.CAUSAL_LM, ...))` → `PeftModelForCausalLM` (qua `mapping_func.get_peft_model` + `MODEL_TYPE_TO_PEFT_MODEL_MAPPING`).
- Official example `examples/gralora_finetuning/gralora_finetuning.py` KHÔNG truyền `task_type` (vẫn chạy, `PeftModel` generic) — lab nên truyền `TaskType.CAUSAL_LM` để có `prepare_inputs_for_generation`/loss đúng chuẩn.

---

## Kết luận

### 1) GraloraConfig có cho phép rank khác nhau từng lớp không? — **KHÔNG**

Bằng chứng trích source v0.21.2:

- `tuners/gralora/config.py`: `GraloraConfig` **không khai báo** trường `rank_pattern`/`alpha_pattern` (so sánh: `tuners/lora/config.py` khai báo cả hai).
- `tuners/gralora/model.py` → `_create_and_replace`:
  ```python
  r = gralora_config.r          # hằng số toàn cục
  ...
  target.update_layer(adapter_name, current_key, r=r, config=gralora_config)
  ```
  — không có bước tra pattern theo `current_key` như `LoraModel` làm.

**Xử lý X2:** dùng r toàn cục; O chỉ phân phối ngân sách theo nhóm mức — chi tiết tại `docs/decisions.md` mục D-X2.

### 2) Ba phát hiện quyết định cho impl-agent

| # | Phát hiện | Ảnh hưởng |
|---|---|---|
| 1 | Khối GraLoRA là **lát cắt** `gralora_A.default[n]` / `gralora_B.default[n]` của 1 param 3D (shape `[k, in_blk, r]` / `[k, r, out_blk]`), A lưu **transposed** so với LoRA chuẩn | X3: đọc grad theo lát `grad[n]`; X4: optimizer state tự đánh chỉ số theo khối (1 param duy nhất) |
| 2 | PiSSA (`init_lora_weights="pissa"`/`"pissa_niter_*"`) **ghi đè** trọng số gốc thành W_res; `from_pretrained` tái lập bằng cách chạy lại init | X1: Oi và PiSSA loại trừ nhau; base-model checkpoint sau PiSSA không còn là W gốc |
| 3 | PEFT **có sẵn** `create_loraplus_optimizer` (mới so với kỳ vọng), group theo substring `"lora_B"` | Baseline LoRA+ dùng được ngay; KHÔNG dùng cho X4 (mức tensor, không theo khối; substring cũng bắt `gralora_B`) |
