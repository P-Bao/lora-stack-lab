# lora-stack-lab

Composability study of LoRA improvements: **do các cải tiến LoRA chồng lên nhau có thực sự cộng hưởng hay không?**

Stack được khảo sát: **GraLoRA** (cấu trúc khối k×k) + **GoRA** (phân bổ rank theo gradient + init) + **AltLoRA** (optimizer chiếu luân phiên) + **BA-LoRA** (3 regularizer độ bền vững), trên nền **một phiên bản `peft` duy nhất ghim cứng**.

> **Trạng thái:** đang xây dựng harness + clean-room reimplementation (xem `docs/reimpl_report.md`).

## Phương pháp triển khai

- `L` (LoRA) và `G` (GraLoRA) dùng **API công khai của PEFT** (`LoraConfig`, `GraloraConfig`).
- `O/Oi` (GoRA), `A` (AltLoRA), `B/Bp/Br` (BA-LoRA regularizers) được **viết lại từ bài báo** (clean-room) trong `src/lorastack/` — hậu tố `-reimpl`; kết quả của các thành phần này **không** tuyên bố tái hiện số liệu chính thức của tác giả nếu chưa qua so sánh hộp đen (`tests/blackbox/`).
- Không fork, không vá gói `peft`. Không vendoring mã nguồn chưa license.

## Cấu trúc

```
src/lorastack/   config, registry, pipeline (thứ tự ghép cố định), adapters, optim, losses, data, eval, utils
spec/            đặc tả rút ra từ paper cho từng phương pháp (nguồn của bản reimpl)
scripts/         run_exp / run_queue (2 GPU song song) / collect_results / aggregate
configs/         base.yaml + experiments/*.yaml (mỗi thí nghiệm một file)
notebooks/       kaggle_runner.ipynb (upload thủ công lên Kaggle, bật Internet)
tests/           test_env.py (ghim phiên bản) + test tính chất + tests/blackbox/
docs/            decisions.md, peft_inventory.md, reimpl_report.md, LICENSES.md, budget.md, interactions.md
```

## Workflow Kaggle (T4 x2, fp16)

1. Upload `notebooks/kaggle_runner.ipynb`, bật Internet, sửa `BRANCH`/`REPO`/`RUN_IDS`.
2. **Save & Run All (Commit)** — kết quả JSON được ghi ngay sau mỗi run; TimeGuard tự lưu checkpoint ở ~10h30 và đóng gói zip.
3. Tải `results_pack.zip` về, giải nén vào nhánh `results/`, chạy `python scripts/aggregate.py` trên máy local.

## Credits & Licenses

| Thành phần | Nguồn | License |
|---|---|---|
| LoRA, GraLoRA (qua PEFT) | [huggingface/peft](https://github.com/huggingface/peft) | Apache-2.0 |
| GraLoRA (bản gốc) | [SqueezeBits/GraLoRA](https://github.com/SqueezeBits/GraLoRA) | Apache-2.0 |
| GoRA | [hhnqqq/MyTransformers](https://github.com/hhnqqq/MyTransformers) | chưa có license — chỉ tham khảo ý tưởng từ paper, viết lại clean-room |
| AltLoRA | [LucasXinYu/AltLoRA](https://github.com/LucasXinYu/AltLoRA) | chưa có license — clean-room |
| BA-LoRA | [llm172/BA-LoRA](https://github.com/llm172/BA-LoRA) | chưa có license — clean-room (PiSSA init lấy từ PEFT) |
| PiSSA | [MuLabPKU/PiSSA](https://github.com/MuLabPKU/PiSSA) | MIT (dùng qua PEFT) |
| RaSA | [zwhe99/RaSA](https://github.com/zwhe99/RaSA) | MIT (cơ sở của repo GraLoRA gốc) |
| LLM-Adapters | [AGI-Edgerunners/LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) | Apache-2.0 (cơ sở của repo GraLoRA gốc) |

Repo này không phân phối lại mã của các repo chưa license; mọi đặc tả và mã reimpl được viết từ các bài báo dưới đây.

## Citation

Nếu repo này hữu ích, vui lòng trích dẫn các paper gốc:

```bibtex
@article{jung2025gralora,
      title={GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning},
      author={Yeonjoon Jung and Daehyun Ahn and Hyungjun Kim and Taesu Kim and Eunhyeok Park},
      year={2025},
      eprint={2505.20355},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20355}
}

@inproceedings{he2025gora,
      title={Go{RA}: Gradient-driven Adaptive Low Rank Adaptation},
      author={Haonan He and Peng Ye and Yuchen Ren and Yuan Yuan and Luyang Zhou and Shucun Ju and Lei Chen},
      booktitle={Advances in Neural Information Processing Systems 38 (NeurIPS 2025)},
      year={2025},
      url={https://arxiv.org/abs/2502.12171}
}

@article{altlora2025,
      title={Towards Better Gradient Approximation in Low-Rank Adaptation with Alternating Projections},
      author={{AltLoRA authors}},
      year={2025},
      eprint={2505.12455},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.12455},
      note={NeurIPS 2025}
}

@inproceedings{chang2026balora,
      title={{BA}-Lo{RA}: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models},
      author={Yupeng Chang and Yi Chang and Yuan Wu},
      booktitle={The Fourteenth International Conference on Learning Representations},
      year={2026},
      url={https://openreview.net/forum?id=q0X9SiXiRO}
}
```

## License

Phần code harness của repo này: **Apache-2.0**. Xem `docs/LICENSES.md` cho chi tiết nguồn gốc từng thành phần.
