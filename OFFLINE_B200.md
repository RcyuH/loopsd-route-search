# Chạy route search trên server B200 offline bằng Python/pip

## 1. Môi trường mục tiêu

Chọn **venv riêng cho route search**, dùng source NeMo RL trong thư mục bên cạnh.
Các lệnh bên dưới không cần `uv`, Internet, tải model hay biên dịch extension
trong lúc inference. Đây là backend HF đồng bộ, **một GPU** qua API NeMo RL;
chưa phải môi trường training OPSD hoặc backend vLLM/Megatron.

| Thành phần | Cấu hình của bộ cài này |
|---|---|
| OS/CPU | Linux; máy chuẩn bị phải cùng kiến trúc CPU và nền OS tương thích server |
| Python | CPython 3.13 thông thường, cùng bản vá trên hai máy; đã có `venv` và `pip` |
| GPU | Một B200 hiển thị được trong môi trường chạy |
| PyTorch | `2.13.0+cu132` — CUDA runtime 13.2 |
| Transformers | `5.16.1` |
| Inference | BF16, SDPA, `cuda:0`, batch size 1 ban đầu |
| NeMo RL source | Commit `bb4f0d85674344d01656efdbf8eb2a7f3f737ca3` |
| NeMo Lens | Commit `4d045c03186b90afdc815db46b671f6a21bc7e98` |

Driver NVIDIA trên server phải hỗ trợ CUDA runtime 13.2 của wheel này. Chốt
driver/runtime với môi trường server trước khi chuẩn bị bundle; `pip` không cài
driver. Phiên bản CUDA hiển thị trong `nvidia-smi` là khả năng của driver;
`torch.version.cuda` mới là CUDA của bản Torch đang dùng. Không cần cài CUDA
toolkit/nvcc riêng cho đường inference dùng wheel này.

Nếu server đã có NeMo container hoạt động tốt trên B200, xem mục 5 để dùng
Python của container đó. Không chép `.venv` từ Mac sang Linux.

## 2. Chuẩn bị trên máy Linux có Internet

Máy chuẩn bị không bắt buộc có GPU nhưng phải cùng CPU architecture (x86_64
hoặc aarch64), Python và OS/container baseline với server. Nếu chỉ có Mac,
dùng máy Linux hoặc Linux container đúng kiến trúc; không chạy script trực tiếp
trên macOS. Cài sẵn Python 3.13, venv, pip, Git, CA certificates và công cụ build
cơ bản trên máy chuẩn bị. Server cần Python/venv/pip và Git được chuẩn bị trước.

Giữ layout khi chuyển dữ liệu; `/srv/loopsd` dưới đây chỉ là đường dẫn ví dụ:

```text
/srv/loopsd/project/
├── code/
│   ├── nemo-rl/                  # Bao gồm .git để ghi provenance
│   └── loopsd-route-search/
│       ├── offline-bundle/       # Sinh bởi script dưới đây
│       ├── configs/
│       ├── examples/
│       ├── scripts/
│       └── src/
├── models/Qwen3-1.7B/            # Trọng số thật, config và tokenizer
└── data/gsm8k/
    ├── train-00000-of-00001.parquet
    └── test-00000-of-00001.parquet
```

Trên máy Linux có mạng, từ thư mục code đã sao chép:

```bash
cd /srv/loopsd/project/code/loopsd-route-search
PYTHON=python3.13 bash scripts/prepare_offline.sh
```

Script tạo `.venv-prepare/` mới, tải Torch từ CUDA index chính thức, build wheel
Lens đúng revision, và gom mọi dependency vào `offline-bundle/wheelhouse/`.
Sau đó cài từ wheelhouse để xuất **requirements.lock.txt gồm cả dependency gián
tiếp**, thông tin môi trường và SHA256. `requirements.txt` trong repo chỉ khóa
dependency trực tiếp; file lock trong bundle là file dùng để cài trên server.
Quá trình này tải nhiều GB; dành đủ dung lượng cho wheelhouse và venv.

Script từ chối ghi vào bundle/venv đã tồn tại để không trộn nhiều lần resolve.
Nếu cần chuẩn bị lại, dùng tên mới:

```bash
BUNDLE="$PWD/offline-bundle-v2" PREP_VENV="$PWD/.venv-prepare-v2" \
  PYTHON=python3.13 bash scripts/prepare_offline.sh
```

Chuyển **offline-bundle**, source `loopsd-route-search`, checkout `nemo-rl` gồm
`.git`, model và parquet sang server bằng cách truyền file bạn đang dùng.
Không cần chuyển CRISP, `.venv` của Mac hoặc outputs chạy trên Mac. Nếu model
có symlink tới Hugging Face cache, chuyển cả nội dung đích; Git LFS pointer
không thay thế được file trọng số. Python interpreter, driver và thư viện OS
không nằm trong wheelhouse: chuẩn bị chúng trong OS/container trước.

## 3. Cài trên server không có mạng

```bash
cd /srv/loopsd/project/code/loopsd-route-search
python3.13 -m venv .venv-b200
source .venv-b200/bin/activate

# Đối chiếu environment.json với Python/CPU/OS server trước khi cài.
cat offline-bundle/environment.json
(cd offline-bundle && sha256sum -c SHA256SUMS)
python -m pip install --no-index --find-links=offline-bundle/wheelhouse --upgrade pip
python -m pip install --no-index --find-links=offline-bundle/wheelhouse \
  -r offline-bundle/requirements.lock.txt
python -m pip check

source scripts/offline_env.sh
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())'
```

Không cần `pip install -e .` hoặc cài package `nemo-rl`: wrapper trong mục 4
thêm source của hai project vào đường dẫn import. Bộ requirements chỉ đủ cho
route search và API imports của checkout NeMo này. Không dùng nó để thay thế
dependency của toàn bộ NeMo training stack.

## 4. Lệnh chạy

Cấu hình server là `configs/qwen3_1p7b_gsm8k_b200.yaml`: ép CUDA, giữ budget
400+128 và protocol search ban đầu, output riêng `outputs/qwen3_1p7b_gsm8k_b200`.
Đường dẫn model/data/NeMo được tính từ **file YAML**, nên layout trên có thể
được đặt ở bất kỳ mount point nào. Nếu đổi layout, sửa các path trong YAML
trước lệnh đầu tiên.

Từ thư mục `loopsd-route-search`, sau khi activate venv và source offline_env:

```bash
# Xem split và số lượt sinh, chưa load trọng số.
python examples/nemo_search.py plan --config configs/qwen3_1p7b_gsm8k_b200.yaml

# Kiểm tra route/cache trên checkpoint thật ở GPU đích trước full search.
python examples/nemo_search.py verify --config configs/qwen3_1p7b_gsm8k_b200.yaml --verify-route 8:12

# Baseline 128 câu của stage 1, được tái sử dụng khi search.
python examples/nemo_search.py baseline --config configs/qwen3_1p7b_gsm8k_b200.yaml

# Chạy cả stage 1, stage 2 và validation để khóa route.
python examples/nemo_search.py search --config configs/qwen3_1p7b_gsm8k_b200.yaml

# So route đã chọn với standard trên toàn bộ GSM8K test.
python examples/nemo_search.py evaluate --config configs/qwen3_1p7b_gsm8k_b200.yaml
python examples/nemo_search.py report --config configs/qwen3_1p7b_gsm8k_b200.yaml
```

Để dừng ở stage 1, thêm `--stop-after stage1` vào lệnh search. Chạy lại lệnh
search đầy đủ để tiếp tục. Sau khi mở shell mới, activate lại venv, source
offline_env và đặt CUDA_VISIBLE_DEVICES trước khi resume. Giữ nguyên code,
config, môi trường, đường dẫn và output của run đó.

Có thể đứng ngay trong `nemo-rl` sau khi activate cùng venv:

```bash
cd /srv/loopsd/project/code/nemo-rl
source ../loopsd-route-search/scripts/offline_env.sh
export CUDA_VISIBLE_DEVICES=0
python ../loopsd-route-search/examples/nemo_search.py search \
  --config ../loopsd-route-search/configs/qwen3_1p7b_gsm8k_b200.yaml
```

Batch size 1 là điểm bắt đầu bảo thủ, chưa được tối ưu cho B200. Có thể tăng
`model.batch_size` sau khi đo bộ nhớ; khi đổi batch size, dùng `output_dir` mới.
Route lặp toàn bộ 28 lớp có KV cache dài nhất; route ngắn chạy vừa chưa đảm bảo
mọi route đều vừa. Nhiều B200 không tự động tăng tốc backend một GPU này.

## 5. Nếu đã có môi trường NeMo RL chạy được trên B200

Dùng `python` của môi trường đó với cùng wrapper và config ở mục 4. Không chạy
`pip install -r requirements.txt` vào môi trường training hiện hữu vì nó có thể
thay Torch/Transformers và phá ABI của các extension.

Trên bản sao container/môi trường NeMo **có mạng**, bổ sung các thư viện thiếu
theo `pyproject.toml` của project này, giữ constraints của recipe NeMo đang dùng.
Route search khai báo Torch `>=2.11,<2.14`, Transformers `>=5.5,<=5.16.1`;
bản đã chạy local là Torch 2.13.0 / Transformers 5.16.1. Khoảng khai báo không
có nghĩa mọi tổ hợp đều đã được kiểm tra. Nếu dependency xung đột, dùng venv
riêng theo mục 2–3. Chuẩn bị xong thì chuyển nguyên container phù hợp với
runtime server; khi chạy dùng Python trực tiếp, không cần trình quản lý uv.

## 6. Phạm vi xác nhận

Code đã được chạy với model local trên Mac/MPS trong lần triển khai trước;
chưa chạy CUDA/B200 và chưa chạy quy trình tạo wheelhouse Linux trong phiên này.
Full search và kết luận looped teacher tốt hơn standard vẫn cần kết quả server.
Requirements cho training OPSD sẽ cần chuẩn bị riêng khi triển khai giai đoạn đó.

Nguồn về cài đặt: [PyTorch 2.13 CUDA wheels](https://pytorch.org/get-started/previous-versions/),
[pip wheel và cài offline](https://pip.pypa.io/en/stable/cli/pip_wheel/).
