# plan.md — Kiểm thử tổ hợp LoRA: GraLoRA + GoRA + AltLoRA (+ BA-LoRA)

> Tài liệu này dành cho **agent điều phối** (ví dụ Claude Code) thực thi trên máy của người dùng.
> Mục tiêu: xác định xem việc **chồng** các cải tiến LoRA có thực sự cộng hưởng hay không, đi từ lõi chính (3 phương pháp) đến thêm phương pháp thứ 4.
> Tài nguyên chạy thử: **Kaggle free, GPU T4 x2, tối đa 12 giờ/session**. Mọi lần chạy đều **clone repo public theo nhánh** (không cần token) rồi chạy, không phụ thuộc trạng thái máy cũ. Notebook được upload và chạy thủ công; kết quả được tải về thủ công.

---

## 0. Quy tắc chung (đọc trước khi làm bất cứ gì)

1. **Không đoán đường dẫn.** Ở bước 0.1 phải hỏi người dùng đường dẫn local của 4 codebase và URL remote Git. Chưa có đủ thì dừng và hỏi.
2. **Repo là PUBLIC, notebook clone không cần token.** Vì vậy tuyệt đối **không commit** khóa, token, file `.env`, đường dẫn riêng tư hay dữ liệu cá nhân. Kiểm tra `git status` và `git diff --cached` trước mỗi lần push.
2b. **Phải push nhánh lên GitHub trước khi chạy trên Kaggle**, vì notebook chỉ lấy được code đã nằm trên remote.
3. **Mỗi phương pháp đã tune learning rate mới được so sánh.** Có bằng chứng rằng LR không phù hợp tạo ra "tiến bộ giả" (bài *Learning Rate Matters*, ICML 2026). Mọi cấu hình, kể cả LoRA gốc, đều phải qua quét LR riêng.
4. **Cổng quyết định (gate)** ở cuối mỗi giai đoạn: không đạt thì dừng, ghi lý do vào `docs/decisions.md`, không cố chồng thêm.
5. **Mọi run phải tái lập được:** lưu seed, commit hash, config đầy đủ, phiên bản thư viện vào file kết quả.
6. Không sửa trực tiếp mã trong `third_party/`. Mọi chỉnh sửa tích hợp nằm trong `src/lorastack/` (bọc, không sửa nguồn). Nếu buộc phải vá, lưu patch riêng trong `third_party/<tên>/PATCHES/` kèm lý do.
7. Số liệu ngân sách trong plan là **ước lượng thô**, phải hiệu chỉnh lại sau Giai đoạn 0 bằng số đo thực tế.

---

## 1. Tên gọi tổ hợp và mã cấu hình

**Tên dự án / repo:** `lora-stack-lab`

**Mã thành phần (viết liền bằng dấu `+`):**

| Mã | Thành phần | Trục tác động |
|----|-----------|---------------|
| `L`  | LoRA gốc (đã tune LR) | đối chứng |
| `G`  | GraLoRA | cấu trúc (lưới khối k×k) |
| `O`  | GoRA, **chỉ phần phân bổ rank** | phân bổ rank |
| `Oi` | GoRA đầy đủ (phân bổ rank + khởi tạo B theo gradient) | rank + init |
| `A`  | AltLoRA | bộ tối ưu (chiếu luân phiên + momentum thấp hạng) |
| `B`  | BA-LoRA (3 regularizer; gốc đi kèm khởi tạo PiSSA) | regularizer / độ bền vững |
| `Bp` | BA-LoRA đầy đủ với init PiSSA | biến thể B |
| `Br` | Chỉ 3 regularizer của BA-LoRA, **không** dùng init PiSSA | biến thể B |

**Tên tổ hợp:**
- **TRIAD-LoRA** = `G+Oi+A` (lõi chính, 3 phương pháp)
- **TETRAD-LoRA** = TRIAD + BA-LoRA, có hai nhánh:
  - `TETRAD-P` = `G+O+A+Bp` (bỏ init của GoRA, dùng PiSSA)
  - `TETRAD-R` = `G+Oi+A+Br` (giữ init của GoRA, chỉ lấy regularizer)

**Quy ước ID thí nghiệm:** `E{giai-đoạn}-{mã-tổ-hợp}-{model}-lr{LR}-r{rank}-s{seed}`
Ví dụ: `E4-G+Oi+A-qwen05b-lr2e-4-r32-s1`.

---

## 2. Hạ tầng Kaggle T4 x2 — ràng buộc kỹ thuật

- **T4 không có bf16 gốc.** Dùng base model fp16 + adapter giữ fp32 + `autocast(fp16)` + `GradScaler`. Các phép nghịch đảo giả (GoRA) và chiếu (AltLoRA) tính trong **fp32**, thêm hệ số ridge nhỏ để ổn định.
- **Không dùng FlashAttention-2.** Dùng SDPA mặc định, bật gradient checkpointing nếu thiếu bộ nhớ.
- **2 GPU x 16 GB.** Với model nhỏ, **chạy 2 run độc lập song song (mỗi GPU một run)** hiệu quả hơn DDP. Đặt `CUDA_VISIBLE_DEVICES=0` và `=1` cho hai tiến trình.
- **Model gợi ý:** Qwen2.5-0.5B (sweep nhanh) và Qwen2.5-1.5B hoặc Llama-3.2-1B (kết quả chính). Không đặt mục tiêu 7–8B trên T4.
- **Giới hạn session 12h:** mọi run có `TimeGuard`: ở mốc ~10h30 tự lưu checkpoint, đẩy kết quả, thoát sạch.
- **Chạy dài nên dùng "Save & Run All (Commit)"** thay vì phiên tương tác (phiên tương tác dễ rớt kết nối). Output của commit được giữ lại và có thể gắn làm input cho lần sau để resume.
- **Quota GPU hàng tuần** của tài khoản free có hạn (khoảng 30 giờ/tuần theo hiểu biết của tôi, hãy xác nhận trên trang Kaggle của bạn). Ghi số đo thực vào `docs/budget.md`.
- **Dung lượng `/kaggle/working`** có giới hạn: chỉ lưu adapter (nhỏ) và JSON kết quả, không lưu cả model.
- **Cần bật Internet** trong notebook để `git clone`.

---

## 3. Cấu trúc repo

```
lora-stack-lab/
├── plan.md
├── README.md
├── requirements-kaggle.txt        # ghim phiên bản (torch, transformers, peft, accelerate, datasets...)
├── third_party/                   # CODE GỐC, copy từ máy người dùng (vendored, không dùng submodule)
│   ├── gralora/   (+ UPSTREAM.md: đường dẫn gốc, commit hash, license)
│   ├── gora/
│   ├── altlora/
│   └── balora/
├── src/lorastack/
│   ├── config.py                  # dataclass cấu hình, đọc YAML
│   ├── registry.py                # đăng ký thành phần theo mã (L,G,O,Oi,A,B...)
│   ├── pipeline.py                # ghép thành phần theo THỨ TỰ CỐ ĐỊNH (mục 4)
│   ├── adapters/
│   │   ├── base.py                # interface chung
│   │   ├── vanilla.py
│   │   ├── gralora_wrap.py
│   │   ├── gora_alloc.py
│   │   ├── gora_init.py
│   │   └── pissa_init.py
│   ├── optim/{altlora.py, factory.py}
│   ├── losses/balora_reg.py
│   ├── data/                      # MetaMathQA(subset), Magicoder(subset), GLUE
│   ├── eval/                      # GSM8K, HumanEval, GLUE, robustness
│   └── utils/{timeguard.py, ckpt.py, logging.py, seed.py, env_report.py}
├── configs/
│   ├── base.yaml
│   └── experiments/               # mỗi thí nghiệm một YAML
├── scripts/
│   ├── bootstrap_kaggle.sh        # clone theo nhánh + cài đặt
│   ├── run_exp.py                 # chạy 1 thí nghiệm theo ID/YAML
│   ├── run_queue.py               # xếp hàng nhiều run, chia 2 GPU, bỏ qua ID đã xong
│   ├── collect_results.py         # gom JSON + manifest thành 1 file zip để tải về
│   └── aggregate.py               # gộp kết quả (trên máy bạn), bảng, hiệu ứng tương tác
├── notebooks/kaggle_runner.ipynb  # MỎNG: bạn upload thủ công; chỉ gọi bootstrap + run_queue + collect
├── tests/
└── docs/{decisions.md, budget.md, interactions.md}
```

---

## 4. Thứ tự ghép thành phần (hợp đồng giao diện)

`pipeline.py` luôn áp dụng theo đúng thứ tự sau, bất kể tổ hợp nào:

1. **Phân bổ rank** (`O`/`Oi`): tính gradient đầy đủ trên N batch, cấp rank `r_i` cho từng lớp theo ngân sách tham số. Chạy streaming từng lớp, giải phóng ngay để tránh giữ cả bản sao gradient của model trên GPU.
2. **Bọc cấu trúc** (`G`): chia mỗi adapter thành lưới k×k khối, rank mỗi khối `r_i/k`.
3. **Khởi tạo**: `Oi` (B theo gradient, theo **từng khối** nếu có `G`) hoặc PiSSA (nếu có `Bp`) hoặc mặc định LoRA.
4. **Bộ tối ưu** (`A`): thay `optim_factory`, áp dụng phép chiếu luân phiên **theo từng khối** nếu có `G`.
5. **Loss** (`B`/`Bp`/`Br`): cộng các regularizer vào loss huấn luyện.

Mỗi thành phần chỉ được chạm vào đúng tầng của mình. Tầng nào cần dữ liệu từ tầng khác thì đọc qua `AdapterContext` (rank từng lớp, kích thước khối, v.v.), không import chéo.

### Các điểm xung đột đã biết (phải xử lý có ghi chép trong `docs/decisions.md`)

| # | Xung đột | Hướng xử lý dự kiến |
|---|----------|--------------------|
| X1 | **`Oi` và PiSSA (trong `Bp`) cùng khởi tạo**; PiSSA sửa trọng số gốc (W_res), GoRA thì không chạm W₀ | Không bao giờ bật cùng lúc. Tách thành TETRAD-P và TETRAD-R |
| X2 | **`G` yêu cầu rank mỗi lớp chia hết cho k** (và chiều lớp chia hết cho k); `O` cấp rank khác nhau từng lớp | Quy tắc làm tròn lên bội số của k, hiệu chỉnh lại ngân sách tham số, ghi lại độ lệch ngân sách |
| X3 | **`Oi` với `G`:** init gradient phải tính trên từng khối con của G | Cài đặt init theo khối, có unit test so với init toàn ma trận khi k=1 (phải trùng) |
| X4 | **`A` với `G`:** AltLoRA thiết kế cho cặp (A,B) nguyên khối | Áp dụng chiếu theo từng khối. Unit test: k=1 phải trùng kết quả AltLoRA gốc |
| X5 | **`A` và `Oi` chồng chéo mục đích** (cùng xấp xỉ gradient đầy đủ, một ở init, một xuyên suốt) | Đo bằng ablation ở Giai đoạn 3, báo cáo cộng hưởng hay dư thừa |
| X6 | **`B` nhắm mục tiêu khác** (độ bền vững/nhiễu), không phải độ chính xác thuần | Đánh giá riêng bằng bộ chỉ số bền vững (Giai đoạn 5) |

**Quy tắc kiểm thử đơn vị bắt buộc:** với `k=1` và rank đồng đều, mỗi thành phần phải tái hiện (trong sai số số học) kết quả của code gốc trên một batch cố định.

---

## 5. Chiến lược Git và vòng đời một lần chạy trên Kaggle (repo public, tải kết quả thủ công)

### 5.1 Nhánh

| Nhánh | Mục đích |
|-------|----------|
| `main` | Harness ổn định, chỉ merge khi qua test + smoke test |
| `stage0/harness` | Hạ tầng, T4 smoke test |
| `stage1/baseline` | LoRA gốc + quét LR/rank |
| `stage2/single-g`, `stage2/single-o`, `stage2/single-a`, `stage2/single-b` | Tích hợp riêng từng phương pháp (4 nhánh, làm song song) |
| `stage3/pair-go`, `stage3/pair-ga`, `stage3/pair-oa` | Các cặp |
| `stage4/triad` | TRIAD-LoRA |
| `stage5/tetrad-p`, `stage5/tetrad-r` | Thêm BA-LoRA, hai nhánh |
| `stage6/final` | Chạy cuối nhiều seed, viết kết quả |
| `results` | Nhánh **orphan**, **bạn tự commit** các JSON đã tải về từ Kaggle (tách khỏi code, tránh xung đột merge) |

**Tái lập:** notebook clone theo *tên nhánh*, nhưng nhánh có thể di chuyển sau khi chạy. Vì vậy mỗi file kết quả luôn ghi **commit hash đã chạy**. Muốn tái lập chính xác thì `git checkout <hash>`. Notebook có biến `COMMIT` tùy chọn để ghim đúng commit khi cần.

### 5.2 Khởi động mỗi session Kaggle

Bạn upload `notebooks/kaggle_runner.ipynb` thủ công và bật **Internet** trong notebook. Chỉ cần sửa 3 biến ở ô đầu: `BRANCH`, `REPO`, và `RUN_IDS` (hoặc `STAGE`).

**Mã code clone vào `/tmp/repo`** (không phải `/kaggle/working`), để thư mục output của Kaggle chỉ chứa kết quả và checkpoint nhỏ, không chứa cả repo.

```python
# Ô 1 của notebooks/kaggle_runner.ipynb
import os, subprocess, sys
BRANCH  = "stage2/single-g"        # <- đổi theo giai đoạn
REPO    = "USER/lora-stack-lab"    # <- điền thật (repo public)
COMMIT  = None                     # <- tùy chọn: hash để ghim đúng commit
RUN_IDS = []                       # <- danh sách ID cần chạy; để trống = chạy cả giai đoạn theo configs

sh = lambda c: subprocess.run(c, shell=True, check=True)
sh(f"git clone --depth 1 --branch {BRANCH} https://github.com/{REPO}.git /tmp/repo")
if COMMIT:
    sh(f"cd /tmp/repo && git fetch --depth 1 origin {COMMIT} && git checkout {COMMIT}")
os.chdir("/tmp/repo")
sh("bash scripts/bootstrap_kaggle.sh")   # pip install -r requirements-kaggle.txt, in env_report
sys.path.insert(0, "/tmp/repo/src")
```

```python
# Ô 2: chạy hàng đợi (2 GPU song song), bỏ qua ID đã có kết quả
ids = " ".join(RUN_IDS)
sh(f"python scripts/run_queue.py --stage {BRANCH.split('/')[0]} --ids {ids} "
   f"--out /kaggle/working/results --ckpt /kaggle/working/ckpt "
   f"--skip-from /kaggle/input")    # quét JSON trong output các lần chạy trước (nếu bạn gắn làm input)
```

```python
# Ô 3: đóng gói để tải về
sh("python scripts/collect_results.py --src /kaggle/working/results "
   "--zip /kaggle/working/results_pack.zip")
```

### 5.3 Trong session

1. `run_queue.py` đọc ID cần chạy (`RUN_IDS` hoặc từ `configs/experiments/`), chia ra **2 GPU**. Trước mỗi run nó kiểm tra ID đã có JSON ở `--out` hoặc ở các thư mục `--skip-from` (output lần trước được gắn làm input) và bỏ qua nếu có.
2. Mỗi run xong ghi ngay `results/<ID>.json` vào `/kaggle/working/results/`. Run lỗi ghi `results/<ID>.FAILED.json` kèm traceback (không làm sập cả hàng đợi).
3. `TimeGuard` ở mốc ~10h30: lưu checkpoint adapter + optimizer state (cũng lưu định kỳ ~500 bước), ghi `session_manifest.json`, chạy `collect_results.py` rồi dừng sạch. **Đóng gói kết quả luôn nằm trong luồng thoát, không phụ thuộc ô 3.**
4. **Chạy bằng "Save & Run All (Commit)"** để output được giữ lại sau khi session kết thúc. Chạy tương tác dễ mất dữ liệu nếu rớt kết nối trước khi bạn tải về.
5. Resume run dở: gắn output của lần trước làm input, `run_queue.py` tự tìm `ckpt/<ID>/` trong `/kaggle/input/...` để tiếp tục.

### 5.4 Sau mỗi session, trên máy bạn

1. Tải output của notebook từ Kaggle (file `results_pack.zip`).
2. Giải nén vào repo local, nhánh `results`, thư mục `results/<giai-đoạn>/`.
3. `git add results/ && git commit && git push` (tùy chọn nhưng nên làm để có bản lưu).
4. Chạy `python scripts/aggregate.py` **cục bộ** để cập nhật bảng và hiệu ứng tương tác. Không cần GPU.
5. Cập nhật danh sách ID còn lại cho session tiếp theo (hoặc để `run_queue.py` tự bỏ qua ID đã xong nếu bạn gắn output cũ làm input).

### 5.5 Nội dung bắt buộc của `results/<ID>.json`

`id`, `git_commit`, `branch`, `seed`, `config` đầy đủ, `lr`, `rank`, `k`, `rank_per_layer` (nếu có `O`), số tham số huấn luyện, `train_loss_curve` (thưa), chỉ số đánh giá, thời gian, bộ nhớ đỉnh, phiên bản torch/transformers/peft, tên GPU.

**`session_manifest.json`** (một file mỗi session): danh sách ID đã xong và bị lỗi, branch, commit, GPU, thời điểm bắt đầu/kết thúc, lý do dừng (`done`, `timeguard`, `error`).

---

## 6. Subagent (khi cần song song)

Dùng **git worktree riêng cho mỗi subagent** (`git worktree add ../wt-<tên> <nhánh>`) để tránh đạp lên nhau. Mỗi subagent chỉ được sửa các đường dẫn được cấp.

| Subagent | Việc | Được sửa | Xong khi |
|----------|------|----------|----------|
| `harness-agent` | Giai đoạn 0: khung train/eval, TimeGuard, ckpt, logging, bootstrap | `src/lorastack/{utils,data,eval}`, `scripts/`, `configs/base.yaml`, `notebooks/` | Smoke test T4 qua, `L` tái lập số của LoRA gốc |
| `gralora-agent` | Bọc GraLoRA vào interface | `adapters/gralora_wrap.py`, `third_party/gralora/PATCHES` | Unit test k=1 trùng code gốc |
| `gora-agent` | Tách `O` và `Oi`; streaming gradient | `adapters/gora_*.py` | Unit test: `Oi` trùng init gốc; ngân sách tham số đúng |
| `altlora-agent` | Bọc AltLoRA thành optimizer factory | `optim/` | k=1 trùng AltLoRA gốc trên batch cố định |
| `balora-agent` | Tách 3 regularizer khỏi PiSSA (tạo `Br`) và giữ `Bp` | `losses/balora_reg.py`, `adapters/pissa_init.py` | Hai biến thể chạy được; ghi rõ chi phí thêm mỗi bước |
| `analysis-agent` | `aggregate.py`, bảng, hiệu ứng tương tác | `scripts/aggregate.py`, `docs/` | Bảng + biểu đồ tái sinh được từ nhánh `results` |

**Điều phối:** `harness-agent` xong trước. Sau đó bốn agent tích hợp (`gralora`, `gora`, `altlora`, `balora`) chạy **song song**. Agent điều phối kiểm tra unit test của từng nhánh rồi mới mở Giai đoạn 3. `analysis-agent` có thể bắt đầu sớm trên dữ liệu giả.

---

## 7. Các giai đoạn

### Giai đoạn 0 — Dựng repo và hạ tầng (`stage0/harness`)

- [ ] 0.1 **Hỏi người dùng:** (a) thư mục cha để tạo repo, (b) đường dẫn local của GraLoRA, GoRA, AltLoRA, BA-LoRA, (c) URL remote Git (private), (d) model ưu tiên.
- [ ] 0.2 Tạo thư mục mới `lora-stack-lab/`, `git init`, tạo cấu trúc mục 3, `.gitignore` (loại `results/`, checkpoint, `.env`, cache).
- [ ] 0.3 **Copy** 4 codebase vào `third_party/` (không symlink). Với mỗi cái: ghi `UPSTREAM.md` (đường dẫn gốc, commit hash nếu là git repo, tên license). Kiểm tra license cho phép bundling; nếu không rõ thì cảnh báo người dùng.
- [ ] 0.4 Viết `requirements-kaggle.txt` ghim phiên bản; kiểm tra 4 codebase cùng chạy được với một bộ phiên bản (ghi xung đột vào `docs/decisions.md`).
- [ ] 0.5 Viết khung train/eval, `TimeGuard`, checkpoint/resume, `env_report`, `bootstrap_kaggle.sh`, `collect_results.py`, `run_queue.py` (có `--skip-from`).
- [ ] 0.6 Tạo repo **public** trên GitHub, tạo nhánh orphan `results`, push các nhánh. Quét repo tìm bí mật (token, khóa, đường dẫn riêng) trước khi để public. Thử clone theo nhánh bằng notebook thủ công trên Kaggle (Internet bật).
- [ ] 0.7 **Smoke test trên T4:** 50 bước với `L` trên Qwen2.5-0.5B bằng fp16 + GradScaler; xác nhận không NaN.
- [ ] 0.8 **Đo thông lượng thực tế** (token/s, bộ nhớ đỉnh, thời gian của một run sweep) và điền `docs/budget.md`. Cập nhật mục 8 theo số đo.

**Gate 0:** smoke test qua, clone-theo-nhánh chạy được trên Kaggle, `L` đạt số hợp lý.

### Giai đoạn 1 — Đối chứng LoRA gốc (`stage1/baseline`)

- Tập dữ liệu sweep nhỏ (gợi ý: subset MetaMathQA ~20k mẫu, 1 epoch, seq len 512; đánh giá GSM8K; thêm một tác vụ GLUE nhỏ để chạy nhanh). Tập con và seed cố định, dùng chung cho **mọi** giai đoạn.
- Quét LR cho `L` ở rank 32: `{5e-5, 1e-4, 2e-4, 5e-4, 1e-3}` (1 seed).
- Tại LR tốt nhất, chạy thêm rank `{16, 64}`.
- Chạy 3 seed tại cấu hình tốt nhất để ước lượng **độ lệch chuẩn nền** (σ_L).

**Gate 1:** có `L*` (LR, rank tốt nhất) và σ_L. **Ngưỡng ý nghĩa dùng cho toàn dự án:** một cải tiến chỉ được coi là có tác dụng nếu `mean(cải tiến) − mean(L*) > max(1.0 điểm, 2·σ_L)`. Ghi ngưỡng này vào `docs/decisions.md` **trước khi** xem kết quả các giai đoạn sau.

### Giai đoạn 2 — Tích hợp và tái hiện từng phương pháp riêng (4 nhánh song song)

Mỗi nhánh: tích hợp vào interface → unit test k=1 → quét LR riêng → so với `L*`.

| Nhánh | Cấu hình cần chạy |
|-------|-------------------|
| `stage2/single-g` | `G` với k ∈ {2, 4}, rank 32 và 64 |
| `stage2/single-o` | `O` và `Oi` (tách riêng để biết phần nào đóng góp) |
| `stage2/single-a` | `A` |
| `stage2/single-b` | `Bp` và `Br` |

Mỗi phương pháp quét LR 4 giá trị quanh vùng tối ưu của nó (vùng LR tối ưu của từng biến thể khác nhau).

**Gate 2:** mỗi phương pháp phải (a) qua unit test tái hiện, và (b) đạt **ít nhất ngang** `L*` trong sai số. Phương pháp nào không tái hiện được hoặc tệ hơn rõ rệt thì **dừng lại điều tra** (lỗi tích hợp? LR sai vùng?), không đưa vào bước ghép. Ghi kết luận từng phương pháp. Phương pháp không có lợi riêng lẻ vẫn có thể được giữ nếu có lý do (ví dụ `B` chủ yếu vì độ bền vững), nhưng phải ghi rõ.

### Giai đoạn 3 — Các cặp trong lõi (`stage3/pair-*`)

Chỉ ghép các phương pháp đã qua Gate 2.

| Nhánh | Cặp | Câu hỏi chính |
|-------|-----|---------------|
| `stage3/pair-go` | `G+O`, `G+Oi` | X2, X3: rank không đồng đều với lưới khối; init theo khối |
| `stage3/pair-ga` | `G+A` | X4: chiếu theo khối |
| `stage3/pair-oa` | `Oi+A`, `O+A` | X5: dư thừa hay cộng hưởng giữa init gradient và optimizer gradient |

Mỗi cặp: quét LR 3 giá trị, rank 32, 1 seed; cặp nào vượt ngưỡng thì chạy thêm 3 seed.

**Hiệu ứng tương tác** (tính trong `aggregate.py`, ghi vào `docs/interactions.md`):
`Δ_int(X,Y) = S(X+Y) − S(X) − S(Y) + S(L*)`
Dương = cộng hưởng; xấp xỉ 0 = cộng dồn độc lập; âm = triệt tiêu.

**Gate 3:** ít nhất một cặp có Δ_int không âm và kết quả vượt ngưỡng so với `L*`. Nếu mọi cặp đều triệt tiêu thì **dừng và ghi kết quả âm** (đây vẫn là kết quả có giá trị).

### Giai đoạn 4 — Lõi chính TRIAD-LoRA (`stage4/triad`)

- Chạy `G+Oi+A` và các biến thể ablation `G+O+A` (bỏ init), cùng các cặp đã có làm so sánh.
- Quét LR 3 giá trị; rank {32, 64} (rank cao là nơi `G` và `O` kỳ vọng có lợi).
- Tại cấu hình tốt nhất: 3 seed.
- Dùng thêm **model 1.5B** cho cấu hình tốt nhất, để kiểm tra xu hướng có giữ khi tăng kích thước.
- Tính Δ_int bậc ba (so với tổng hiệu ứng riêng và các cặp) để xem phần cộng hưởng bậc cao.

**Gate 4:** TRIAD vượt rõ rệt cặp tốt nhất (theo ngưỡng mục 7/Gate 1) thì tiếp tục sang Giai đoạn 5. Nếu không, **TRIAD không đáng chồng thêm**: ghi kết luận và chuyển sang viết kết quả bằng cặp tốt nhất.

### Giai đoạn 5 — Thêm phương pháp thứ 4: BA-LoRA (`stage5/tetrad-p`, `stage5/tetrad-r`)

Hai nhánh song song (do xung đột X1):

- **TETRAD-P** = `G+O+A+Bp`: bỏ init của GoRA, dùng PiSSA như BA-LoRA gốc.
- **TETRAD-R** = `G+Oi+A+Br`: giữ init của GoRA, chỉ thêm 3 regularizer.

Đánh giá hai lớp chỉ số:
1. **Độ chính xác** như các giai đoạn trước (so với TRIAD).
2. **Độ bền vững** (BA-LoRA nhắm vào mục tiêu này): so mô hình nền sạch và nền nhiễu hơn như bài gốc nếu có thể, hoặc dùng proxy chèn **nhiễu nhãn** (ví dụ 20%) lên tác vụ GLUE nhỏ. *(Proxy này là thiết kế của plan, không phải phương pháp trong bài gốc, cần ghi rõ khi báo cáo.)*

Đo và báo cáo **chi phí thêm** của regularizer (thời gian/bước, bộ nhớ đỉnh).

**Gate 5:** TETRAD được ghi nhận có ích nếu (a) giữ được độ chính xác của TRIAD trong sai số **và** cải thiện rõ độ bền vững, hoặc (b) cải thiện cả hai. Nếu không, kết luận BA-LoRA nên được dùng độc lập, không ghép.

### Giai đoạn 6 — Chạy cuối và tổng hợp (`stage6/final`)

- Các cấu hình cuối cùng (L*, đơn lẻ tốt nhất, cặp tốt nhất, TRIAD, TETRAD tốt nhất): **3 seed**, tập đánh giá đầy đủ (GSM8K, HumanEval, GLUE), model chính (1.5B/1B).
- Quét rank {16, 32, 64} cho các cấu hình này để xem lợi ích có **phụ thuộc rank** không.
- `analysis-agent` sinh: bảng chính (mean ± std), bảng tương tác Δ_int, biểu đồ theo rank, bảng chi phí (thời gian, bộ nhớ, tham số).
- Viết `docs/findings.md`: kết luận (kể cả kết quả âm), các giới hạn (model nhỏ, T4, tập con dữ liệu), hướng đi tiếp.

**Gate 6 (hoàn tất):** mọi kết luận trong `findings.md` truy ngược được tới ID thí nghiệm, commit và file JSON.

---

## 8. Ngân sách ước lượng (hiệu chỉnh lại sau Giai đoạn 0)

Giả định thô (cần đo): một run sweep (0.5B, ~20k mẫu, 1 epoch) khoảng 30–45 phút; run chính (1.5B/đầy đủ) 2–3 giờ; chạy 2 run song song trên 2 GPU.

| Giai đoạn | Số run (xấp xỉ) | Ghi chú |
|-----------|----------------|---------|
| 0 | 3–5 | smoke test, đo thông lượng |
| 1 | 10 | 7 sweep + 3 seed |
| 2 | ~24 | 4 phương pháp x ~6 |
| 3 | ~12 | 3 cặp |
| 4 | ~8 | TRIAD + ablation + 1.5B |
| 5 | ~8 | hai nhánh TETRAD + đo bền vững |
| 6 | ~18 | 6 cấu hình x 3 seed (một số đã có) |
| **Tổng** | **~80 run** | **~20–30 giờ session ⇒ 1–2 tuần quota free** |

Nếu quota không đủ: (1) bỏ 1.5B, chỉ giữ 0.5B; (2) giảm seed cuối còn 2; (3) bỏ k=4 của GraLoRA; (4) bỏ HumanEval (sinh chậm) và chỉ giữ GSM8K + GLUE. Mọi cắt giảm phải ghi vào `docs/decisions.md`.

---

## 9. Danh sách rủi ro

| Rủi ro | Giảm thiểu |
|--------|-----------|
| fp16 trên T4 gây NaN/overflow (đặc biệt phép nghịch đảo giả, chiếu) | Tính các phép này bằng fp32 + ridge; GradScaler; kiểm tra NaN mỗi bước và dừng sớm |
| Bốn codebase xung đột phiên bản thư viện | Ghim phiên bản ở Giai đoạn 0.4; nếu không thể, bọc bằng tiến trình riêng và ghi vào decisions |
| GoRA tính gradient đầy đủ tốn bộ nhớ | Streaming từng lớp; giảm N batch; đo ở Giai đoạn 0.8 |
| Tích hợp sai khiến kết quả "tệ" oan | Unit test k=1 bắt buộc, Gate 2 |
| Kết quả đẹp chỉ vì LR may mắn | Quy tắc mục 0.3 và ngưỡng ý nghĩa của Gate 1 |
| Repo public làm lộ thông tin riêng hoặc code có license hạn chế | Quét bí mật trước khi public (0.6); kiểm license ở 0.3; không commit `.env`, khóa, dữ liệu cá nhân |
| Mất session giữa chừng hoặc quên tải kết quả | Ghi JSON ngay sau mỗi run, TimeGuard đóng gói zip khi thoát, chạy bằng Save & Run All (Commit), tải output ngay sau session |
| Nhánh đã đổi sau khi chạy nên không tái lập được | JSON ghi commit hash; có biến `COMMIT` để ghim |
| License không cho phép vendoring | Kiểm ở bước 0.3, hỏi người dùng trước khi tiếp tục |

---

## 10. Việc cần người dùng cung cấp trước khi bắt đầu

1. Đường dẫn local của GraLoRA, GoRA, AltLoRA, BA-LoRA.
2. Tên tài khoản/URL repo Git **public** và xác nhận Kaggle đã bật Internet cho notebook (có thể cần xác minh số điện thoại).
3. Model ưu tiên (mặc định: Qwen2.5-0.5B cho sweep, Qwen2.5-1.5B cho kết quả chính).
4. Xác nhận quota GPU thực tế còn lại trong tuần.
