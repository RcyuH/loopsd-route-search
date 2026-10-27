# LoOPSD route search on Qwen3

Tìm đoạn lớp chạy hai lần, giữ nguyên trọng số, rồi so sánh route được chọn với
đường chạy chuẩn trên GSM8K test. Đây là phần **teacher construction** của
[LoOPSD v1](https://arxiv.org/pdf/2610.18162v1#page=13); chưa có distillation/training.

Code nằm trong package riêng bên cạnh checkout NeMo RL. `backend: nemo` đi qua
`LoopedHFGeneration`, một subclass thật của `GenerationInterface` trong checkout
NeMo RL. `backend: hf` dùng cùng engine trực tiếp để kiểm tra trên CPU/MPS.
Cấu hình đầy đủ mặc định dùng `backend: nemo`.
Backend này chạy HF greedy inference đồng bộ, một thiết bị; không phải backend
vLLM/Megatron, không khởi tạo cluster Ray hoặc cơ chế refit cho OPSD.

## Môi trường và tài nguyên

Hướng dẫn server **B200 offline**, bộ requirements và cách chuẩn bị wheelhouse:
**[OFFLINE_B200.md](OFFLINE_B200.md)**. Dùng Python/pip trực tiếp, không cần `uv run`.
`requirements.txt` khóa Torch CUDA 13.2 và các dependency trực tiếp; script
`scripts/prepare_offline.sh` tạo wheelhouse và lock đầy đủ trên Linux có mạng.

Sau khi chuẩn bị môi trường theo hướng dẫn đó:

```bash
source .venv-b200/bin/activate
source scripts/offline_env.sh
export CUDA_VISIBLE_DEVICES=0
python examples/nemo_search.py plan --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Checkpoint và dữ liệu đọc trực tiếp từ các đường dẫn trong YAML, không tải lại.
Đường dẫn tương đối được tính từ **file YAML**. Venv Mac hiện có chỉ dành cho
phát triển local; không chuyển sang Linux.

Model được hỗ trợ: **dense Qwen3, full attention**. Model khác hoặc sliding
attention bị từ chối rõ ràng. `device: auto` ưu tiên CUDA, rồi MPS, rồi CPU.
`dtype`, attention implementation và `batch_size` được cố định trong manifest.
Mac 16 GB dùng được để kiểm tra nhỏ; full search gồm hơn 74 nghìn lượt sinh khi
không có thêm route đồng hạng, nên nên chạy trên GPU CUDA. Tăng batch size theo
bộ nhớ GPU, nhưng dùng thư mục output mới khi đổi cấu hình.

## Quy trình

Kiểm tra chia dữ liệu và khối lượng tính toán, chưa load trọng số:

```bash
python examples/nemo_search.py plan --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Kiểm tra cache, chia sẻ tham số và khôi phục route chuẩn trên checkpoint thật:

```bash
python examples/nemo_search.py verify --config configs/qwen3_1p7b_gsm8k_b200.yaml --verify-route 8:12
```

Chạy baseline trên đúng 128 câu của stage 1, kết quả được dùng lại khi search:

```bash
python examples/nemo_search.py baseline --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Tìm kiếm đầy đủ; có thể dừng sau stage 1 để xem kết quả:

```bash
python examples/nemo_search.py search --config configs/qwen3_1p7b_gsm8k_b200.yaml --stop-after stage1
python examples/nemo_search.py search --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Sau khi route được khóa bằng validation, chạy đánh giá test:

```bash
python examples/nemo_search.py evaluate --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Chạy lại cùng lệnh để tiếp tục; các câu đã lưu sẽ được bỏ qua. File lock ngăn
hai tiến trình cùng ghi một run. Nếu bị dừng giữa batch, batch gốc được phát lại
để giữ cùng padding; chỉ các dòng còn thiếu được ghi thêm. Chỉ dòng cuối ghi dở
mới được phục hồi tự động. Lỗi inference, JSON hỏng ở giữa file hoặc thay đổi
provenance sẽ làm lệnh dừng, không bị đổi thành một câu trả lời sai.

## Dùng NeMo RL

Adapter ở `src/loopsd_search/nemo_adapter.py` có `generate`, lifecycle và dữ liệu
`BatchedDataDict` theo checkout thật. Nó trả token IDs, log-probabilities, độ dài
và cờ truncated với right padding. Chỉ hỗ trợ greedy và checkpoint cố định.
Các thao tác refit/collective chưa hỗ trợ sẽ báo lỗi.

Trong môi trường NeMo RL đã được cài đầy đủ, đứng ở checkout NeMo RL và chạy:

```bash
python ../loopsd-route-search/examples/nemo_search.py plan \
  --config ../loopsd-route-search/configs/qwen3_1p7b_gsm8k_b200.yaml
python ../loopsd-route-search/examples/nemo_search.py search \
  --config ../loopsd-route-search/configs/qwen3_1p7b_gsm8k_b200.yaml
```

Wrapper thêm source package này vào `sys.path` và chọn `backend: nemo`.
Đổi `search` thành `evaluate` để đánh giá với cùng backend. Không chạy một run
HF rồi resume bằng NeMo; dùng output riêng vì backend là một phần provenance.
Dependency để import API NeMo trên máy phát triển được tách trong extra
`nemo-api`; extra này không cài CUDA training stack hoặc vLLM.
Bạn cũng có thể chạy trực tiếp từ package này với venv đã tạo:

```bash
python examples/nemo_search.py search --config configs/smoke_nemo.yaml
python examples/nemo_search.py evaluate --config configs/smoke_nemo.yaml
```

CLI tự thêm `nemo_rl_path` trước khi import adapter; nó dùng source của checkout
bạn cung cấp. `requirements/nemo-api.txt` khóa bản Lens; script chuẩn bị build
wheel từ đúng revision của checkout.
Adapter không được tự đăng ký vào các training recipe có sẵn của NeMo RL.

## Thuật toán và các giả định

- Route `i:j` dùng **chỉ số từ 1, cả i và j đều được bao gồm**, theo paper.
  Ví dụ `2:3` trên model 5 lớp chạy `[1,2,3,2,3,4,5]`. Blog RYS dùng chỉ số từ
  0 và điểm cuối loại trừ; không dùng quy ước đó trong file cấu hình này.
- Với 28 lớp, thử 406 đoạn lặp và một đường chuẩn. Các module cho từng lượt
  chạy chia sẻ đúng cùng đối tượng Parameter, nhưng `layer_idx` khác nhau để
  KV cache riêng. Mỗi batch bắt đầu cache mới; không dùng cache qua route khác.
- Shuffle train bằng `numpy.default_rng(42).permutation`. Lấy 1.000 câu search,
  100 câu validation kế tiếp. ID nguồn từng câu được lưu trong manifest. Paper
  nêu seed nhưng không chỉ rõ thư viện shuffle, nên đây là giả định có ghi nhận.
- Stage 1: 128 câu, giữ top 24 và mọi route đồng hạng ở ngưỡng. Stage 2: 872 câu
  còn lại, xếp theo accuracy trên **872 câu này**, giữ top 5 và đồng hạng. Đường
  chuẩn luôn được giữ. Validation: 100 câu riêng; chuẩn thắng nếu đồng hạng tốt
  nhất. Loop đồng hạng được chọn bằng Python `random.Random(42)` trên ID đã sort.
- Prompt giống nhau giữa chuẩn và loop; chưa dùng biến thể concise LoOPSD†.
  Paper không cung cấp nguyên văn prompt, answer cue hay scorer. Prompt mẫu,
  cue, chính sách dừng và scorer ở đây là reconstruction assumptions; chưa thể
  tuyên bố khớp tuyệt đối thí nghiệm tác giả hoặc hứa baseline bằng 42,30%.
- Thinking dừng ở `</think>` hoặc 400 token. Nếu hết budget, thêm `</think>`;
  tiếp đó thêm answer cue và sinh tối đa 128 token. Nếu EOS xuất hiện trước
  `</think>`, câu trả lời kết thúc và được chấm sai do không có answer stage.
  Token closure/cue được thêm vào được lưu riêng, không tính là token model sinh.
- Scorer chỉ đọc **answer stage**: ưu tiên boxed cuối, tiếp theo marker đáp án,
  cuối cùng là dòng cuối chứa thuần số. Hỗ trợ số âm, thập phân, dấu phân cách
  hàng nghìn và phân số; không lấy bừa số cuối từ chuỗi reasoning.
- Test labels chỉ được đọc trong `evaluate`. Báo accuracy cả hai đường chạy,
  số câu chỉ một bên đúng, paired bootstrap CI, số token và chi phí thời gian.
  Cùng budget token không có nghĩa là cùng FLOPs. Thời gian bao gồm inference
  HF hai stage và padding, không phải benchmark serving tối ưu.
- Nếu validation chọn chuẩn, kết luận của search là chưa tìm được loop hữu ích;
  evaluate sẽ so chuẩn với chính nó và ghi rõ điều này. Không chọn lại route từ test.

RYS là nguồn tham khảo về cấu trúc lặp lớp:
https://dnhkng.github.io/posts/rys/

## Artifacts

Mỗi run lưu trong `output_dir`:

| File | Nội dung |
|---|---|
| `manifest.json` | Cấu hình, SHA256 model/data/code, phiên bản thư viện, commit NeMo, các giả định, split IDs |
| `predictions/<stage>/<route>.jsonl` | Câu hỏi, token IDs, thinking/answer, đáp án trích xuất, đúng/sai và thời gian |
| `stage1_summary.json`, `stage2_summary.json`, `validation_summary.json` | Điểm từng route theo từng stage |
| `selected_route.json` | Route cố định, thứ tự thực thi, validation và checksum selection |
| `test_comparison.json` | So sánh trên test cùng paired bootstrap CI |
| `route_scores.csv`, `route_landscape.png` | Bảng và heatmap search; heatmap không phải test accuracy |
| `verification.json` | Kiểm tra numerical trên model thật |

Model/data/code/config thay đổi sẽ làm resume bị từ chối; tạo output mới cho
thí nghiệm mới. Không chỉnh tay selection để thử nhiều route trên test.

## Kiểm thử và smoke run

Các lệnh pytest/ruff dành cho môi trường phát triển đã cài thêm hai công cụ đó;
bộ requirements B200 không cần chúng để inference.

```bash
PYTHONPATH=src:../nemo-rl python -m pytest -q
python -m ruff check .
python -m ruff format --check .
PYTHONPATH=src python -m loopsd_search.cli search --config configs/smoke.yaml
PYTHONPATH=src python -m loopsd_search.cli evaluate --config configs/smoke.yaml
```

Smoke config chỉ thử route `8:12`, 2 câu search, 1 validation, budget 8+8 token
và 2 câu test. Nó xác nhận đường đi dữ liệu và artifact, **không đo chất lượng
suy luận**. Test số học dùng model Qwen3 nhỏ khởi tạo ngẫu nhiên, so với model
mở rộng độc lập và tính lại toàn prefix không cache. Test NeMo runtime chỉ chạy
khi import được checkout NeMo thật; không thay bằng mock framework.

Tham số `selected_route.json` được thiết kế để dùng lại khi xây frozen teacher
cho OPSD. Việc nối nó vào teacher scoring, rollout synchronization và loss
LoOPSD thuộc giai đoạn huấn luyện tiếp theo.
