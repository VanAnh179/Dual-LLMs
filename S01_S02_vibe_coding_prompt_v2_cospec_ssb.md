# Vibe-Coding Prompt — S01 + S02 (SSB Fast Track, Minimal Coupling v1) — v2

> **Trước khi dùng prompt này:** chạy `setup_cospec_ssb_repo.ps1` trên máy Windows để
> đổi tên repo cũ và tạo/populate repo mới `cospec-ssb`. Prompt bên dưới giả định
> script đó đã chạy xong thành công.
>
> Copy toàn bộ khối text bên dưới (từ `Bạn là senior ML engineer...` đến hết) và paste
> vào IDE agent (Claude Code / Cursor / Codex...).

---

````text
Bạn là senior ML engineer + AI researcher. Hãy làm việc trong repo:

D:\Program\Dual LLMs\cospec-ssb

Đây là repo MỚI, tách riêng khỏi repo cũ D:\Program\Dual LLMs\gsm8k-text-collab-baselines
(repo cũ chứa toàn bộ code D11.x/D12.x — chỉ dùng để THAM CHIẾU nếu cần đọc lại logic
cũ, TUYỆT ĐỐI KHÔNG sửa/ghi bất cứ gì vào repo cũ). Mọi file bạn tạo ra đều nằm trong
D:\Program\Dual LLMs\cospec-ssb.

================================================================
PHẦN 0 — KIỂM TRA SETUP TRƯỚC KHI BẮT ĐẦU (BẮT BUỘC LÀM ĐẦU TIÊN)
================================================================

Trước khi viết bất kỳ code nào, kiểm tra repo mới đã được setup đúng chưa:
  1. data/raw/train.jsonl và data/raw/test.jsonl tồn tại và không rỗng.
  2. src/ có đủ: data_utils.py, generation.py, evaluation.py, answer_extraction.py,
     prompts.py, training.py.
  3. scripts/train_alternating_lora.py tồn tại (dùng để fallback retrain nếu cần).
  4. outputs/imported_d11_adapters/agent_A_round_1 và .../agent_B_round_1 tồn tại.

Nếu THIẾU bất kỳ mục nào ở trên, DỪNG LẠI ngay, in rõ danh sách file/thư mục còn thiếu,
và yêu cầu tôi chạy lại setup_cospec_ssb_repo.ps1 trước — KHÔNG tự ý tạo lại các file
này bằng cách đoán nội dung.

================================================================
PHẦN 1 — BỐI CẢNH & MỤC TIÊU THÍ NGHIỆM
================================================================

Đây là bước đầu tiên của track "SSB Fast Track" (mã: S01, S02) trong dự án nghiên cứu
CoSpec-SSB — Selective State-Space Communication Bus cho hệ Dual LLM. Mục tiêu track
này là kiểm tra NHANH xem một kênh latent giữa hai Qwen2.5-1.5B-Instruct có tạo ra
tín hiệu communication thật hay không, TRƯỚC KHI đầu tư công sức xây Mamba đầy đủ hoặc
dataset mới. Vì vậy S01+S02 chạy ngay trên GSM8K + adapter D11.0 đã copy sẵn vào repo
này, KHÔNG cần dataset mới, KHÔNG cần Mamba ở bước này.

Baseline đã có (dữ liệu tham chiếu, đến từ repo cũ, đã copy vào đây):
  outputs/imported_d11_adapters/agent_A_round_1
  outputs/imported_d11_adapters/agent_B_round_1
  D11.0: agent_A_alone_accuracy=0.58, agent_B_alone_accuracy=0.65, A_then_B_accuracy=0.68
  D12.0: majority_vote_accuracy=0.50 (đã bác bỏ giả thuyết ensembling thuần tuý)

Ý tưởng kỹ thuật S02 (Minimal Latent Coupling v1 — CHƯA phải Mamba, chỉ là phép chiếu
tuyến tính tối giản để test cơ chế trước):
  1. Agent A (giữ nguyên adapter đã import, ĐÓNG BĂNG hoàn toàn, không train) forward
     qua full problem text (KHÔNG split — dùng nguyên GSM8K problem), trích hidden
     state tại 1 layer giữa stack (layer index lấy từ config, mặc định 14 trên 28
     layers).
  2. Mean-pool hidden state đó qua chiều thời gian -> 1 vector -> linear projection
     xuống d_bottleneck=64 -> vector z. Bước này (Write Encoder) CÓ trainable params.
  3. z được chiếu ngược lại d_model qua 1 linear layer, kết hợp vào hidden state của
     Agent B tại cùng layer index qua 1 gate học được (Gated Residual Injection):
       h_B_new = h_B + sigmoid(gate([h_B, z_proj])) * z_proj
  4. Agent B (LoRA train tiếp từ adapter đã import hoặc fresh — parameterize qua
     config) sinh ra final answer, loss SFT cross-entropy CHỈ trên output của B
     (Agent A không sinh text nào trong nhánh này, chỉ dùng để lấy hidden state qua
     torch.no_grad()).

Đây KHÔNG phải experiment forced-cooperation (A và B vẫn cùng thấy full problem) —
mục đích DUY NHẤT của S02 là kiểm tra cơ chế kỹ thuật: hook đúng layer, gradient chảy
được qua bridge/gate, pipeline train/eval chạy ổn định, và có 1 con số accuracy đầu
tiên để so với D11.0 text pipeline (0.68). Causal diagnostic (shuffle/zero/noise test
để xác nhận B có THỰC SỰ dùng z hay không) là task riêng (S03), KHÔNG nằm trong scope
prompt này.

================================================================
PHẦN 2 — CHỈ THỊ BẮT BUỘC (CRITICAL EXECUTION RULES)
================================================================

1. TỰ ĐỘNG CHẠY THỬ (SMOKE TEST) BẰNG GPU: Sau khi viết xong CODE CHO MỖI SCRIPT, bạn
   PHẢI dùng công cụ Terminal của mình để chạy thử NGAY trên GPU với cấu hình cực nhỏ:
     - Script kiểm tra hook/forward: chạy với đúng 1 mẫu.
     - Script train: chạy với `--max-examples 2` (hoặc field config tương đương) và
       giới hạn `max_steps` rất nhỏ (2-4 step) chỉ để xác nhận loss.backward() chạy
       được, không NaN, gradient của bridge/gate params khác None sau backward.
     - Script eval: chạy với `--max-examples 2`.
   TUYỆT ĐỐI không được dừng lại ở việc "viết xong code" mà chưa chạy thử.

2. TỰ SỬA LỖI, KHÔNG HỎI NGƯỢC LẠI: Nếu smoke test lỗi (import lỗi, shape mismatch,
   CUDA OOM, NaN loss, gradient None...), bạn PHẢI tự phân tích và sửa code, chạy lại,
   lặp lại đến khi smoke test pass. KHÔNG được hỏi tôi "bạn đã chạy thử chưa" hay dừng
   lại chờ xác nhận — bạn phải tự chạy trước khi báo cáo.

3. CHỈ SAU KHI SMOKE TEST PASS mới được dừng lại và in ra các câu lệnh Bash CHÍNH XÁC
   để tôi tự chạy full run (full GSM8K train split cho training, 100 mẫu first_n cho
   eval). Không tự ý chạy full run thay tôi (full run tốn thời gian/compute, để tôi
   chủ động chạy).

4. DATA GUARDS: dùng reject_test_split_for_training() và reject_test_rows_for_training()
   trong mọi script train; dùng reject_train_split_for_final_eval() và
   reject_train_rows_for_final_eval() trong script eval; gọi record_sampled_ids().
   Tái dùng các hàm này từ src/data_utils.py (đã copy vào repo này), KHÔNG viết lại.

5. KHÔNG ĐƯỢC ĐỘNG VÀO REPO CŨ: mọi thao tác đọc/ghi chỉ diễn ra trong
   D:\Program\Dual LLMs\cospec-ssb. Nếu cần xem lại logic cũ, chỉ ĐỌC file trong
   D:\Program\Dual LLMs\gsm8k-text-collab-baselines, không sửa, không ghi đè.

6. KHÔNG dùng placeholder lười biếng kiểu `# TODO: implement...`. Code phải chạy được
   thật, không phải khung sườn.

7. BÁO CÁO TỔNG HỢP LÀ BẮT BUỘC (ưu tiên cao, không được bỏ qua): sau khi toàn bộ S01
   và S02 chạy xong (bao gồm cả phần eval), bạn PHẢI viết 1 file báo cáo tổng hợp đầy
   đủ theo đúng PHẦN 5 bên dưới — không chỉ ghi số liệu khô mà phải có đủ: hướng đi
   (vì sao làm S01/S02 trước SSB đầy đủ), phương pháp (kiến trúc Minimal Coupling hoạt
   động thế nào), cách triển khai (những quyết định kỹ thuật cụ thể đã chọn: layer nào,
   bottleneck dim bao nhiêu, đóng băng Agent A vì sao), và kết quả (bảng số liệu + đọc
   kết quả ban đầu). Đây là input trực tiếp cho quyết định Gate S03-04 sau này.

================================================================
PHẦN 3 — CẤU TRÚC FILE (Sử dụng CHÍNH XÁC tên file này, tất cả trong cospec-ssb)
================================================================

Nhóm S01 (Hạ tầng):
  scripts/S01_verify_adapter_loadable.py
  scripts/S01_test_mamba_forward_smoke.py
  src/S01_hook_utils.py
  src/S01_sequential_runner.py

Nhóm S02 (Minimal Latent Coupling v1):
  configs/s02_minimal_coupling_gsm8k.yaml
  src/S02_minimal_coupling.py
  scripts/S02_test_forward_single_example.py
  scripts/S02_train_minimal_coupling.py
  scripts/S02_evaluate_minimal_coupling.py

Report tổng hợp (bắt buộc, xem PHẦN 5):
  notes/S01_S02_ssb_prototype_report.md
  notes/S02_minimal_coupling_results.md  (bản ngắn, đúng format chuẩn D11.x/D12.x)

================================================================
PHẦN 4 — IMPLEMENTATION PHASES
================================================================

--- NHÓM S01 ---

PHASE S01-1 — scripts/S01_verify_adapter_loadable.py
  - Thử load outputs/imported_d11_adapters/agent_A_round_1 và
    outputs/imported_d11_adapters/agent_B_round_1 bằng peft.PeftModel.from_pretrained()
    trên base Qwen/Qwen2.5-1.5B-Instruct.
  - Nếu load thành công: chạy 1 forward pass tối giản (1 câu GSM8K bất kỳ từ
    data/raw/train.jsonl) để xác nhận adapter hoạt động (output không rỗng, không NaN).
    In "ADAPTER_STATUS: OK" và dừng — KHÔNG cần retrain.
  - Nếu load lỗi (kể cả lỗi do file bị cắt cụt như ghi nhận trong D12_0_voting_report.md
    của repo cũ — dấu hiệu: dung lượng file bất thường nhỏ, xem cảnh báo mà
    setup_cospec_ssb_repo.ps1 đã in ra lúc copy): in rõ lỗi, in hướng dẫn "cần chạy:
    python scripts/train_alternating_lora.py (1 Round)" — LƯU Ý script này giờ chạy
    NGAY TRONG repo cospec-ssb (đã copy sẵn ở scripts/train_alternating_lora.py), output
    adapter mới sẽ lưu vào outputs/S01_baseline_retrained/adapters/ (KHÔNG ghi đè vào
    outputs/imported_d11_adapters/ để phân biệt rõ "bản gốc import" và "bản retrain tại
    đây"). Dừng lại KHÔNG tự động retrain thay tôi — để tôi chủ động quyết định.

PHASE S01-2 — scripts/S01_test_mamba_forward_smoke.py
  - Cài đặt (nếu chưa có) mamba-ssm >= 2.0.3 bằng
    `pip install mamba-ssm --break-system-packages`.
  - LƯU Ý: mamba-ssm cần Triton + CUDA, có thể gặp vấn đề trên native Windows. Nếu môi
    trường hiện tại là Windows và cài đặt/import lỗi do thiếu Triton kernel hỗ trợ
    Windows, in rõ thông báo lỗi cụ thể + gợi ý dùng WSL2 hoặc container Linux, rồi
    DỪNG LẠI (không phải lỗi cần bạn tự sửa bằng cách viết lại kernel). Việc này KHÔNG
    chặn S02 (S02 không cần mamba-ssm).
  - Nếu cài đặt thành công: tạo 1 `mamba_ssm.Mamba` block nhỏ (d_model=64), forward 1
    tensor ngẫu nhiên shape (1, 10, 64) qua GPU, in shape output để xác nhận kernel
    chạy được. Đây chỉ là smoke test chuẩn bị cho S04 sau này, không dùng trong S02.

PHASE S01-3 — src/S01_hook_utils.py
  - Viết hàm `get_layer_by_index(model, layer_idx)` để lấy đúng decoder layer thứ
    `layer_idx` của Qwen2.5-1.5B (dùng `model.model.layers[layer_idx]`, verify bằng
    cách in `type(layer)` và `layer_idx` hợp lệ trong khoảng [0, num_hidden_layers)
    lấy từ `model.config.num_hidden_layers`, KHÔNG hardcode số 28).
  - Viết class `HiddenStateExtractor` dùng `register_forward_hook` để lưu lại
    `hidden_states` output của layer chỉ định vào 1 buffer instance attribute mỗi lần
    forward, có `.remove()` để gỡ hook khi xong.
  - Viết class `HiddenStateInjector` dùng `register_forward_pre_hook` trên layer chỉ
    định để CHỈNH SỬA input hidden_states trước khi layer đó xử lý tiếp (dùng để chèn
    injection ở S02). Injection function truyền vào dưới dạng callable
    `injection_fn(original_hidden_states) -> new_hidden_states`.
  - Viết docstring rõ ràng, vì đây là utility dùng lại nhiều lần ở S02/S04.

PHASE S01-4 — src/S01_sequential_runner.py
  - Viết hàm `run_sequential(load_a_fn, forward_a_fn, load_b_fn, forward_b_fn,
    clear_between=True)` — load A, chạy forward_a_fn, lưu kết quả cần thiết (hidden
    state đã detach lên CPU nếu clear_between=True), gọi clear_model() (tái dùng từ
    src/ đã copy) để giải phóng VRAM, rồi mới load B và chạy forward_b_fn.
  - Đây là runner dùng chung cho S02 khi cần tiết kiệm VRAM lúc eval (không cần cho
    training vì lúc train cả A và B đều phải ở trên GPU cùng lúc để gradient chảy qua
    injection — chỉ dùng runner này khi A bị đóng băng và chỉ cần trích xuất z một lần
    rồi cache, xem PHASE S02-4).

--- NHÓM S02 ---

PHASE S02-1 — configs/s02_minimal_coupling_gsm8k.yaml
  Theo đúng cấu trúc UNIVERSAL_EXPERIMENT_TEMPLATE.md, thêm section đặc thù:
    minimal_coupling:
      layer_index: 14          # cả extract ở A và inject ở B dùng chung layer này
      bottleneck_dim: 64
      freeze_agent_a: true      # Agent A giữ nguyên adapter import, KHÔNG train
      init_agent_b_from_d11: true   # true = tiếp tục từ adapter B import, false = fresh LoRA
      agent_a_adapter_path: outputs/imported_d11_adapters/agent_A_round_1
      agent_b_adapter_path: outputs/imported_d11_adapters/agent_B_round_1
    Các field khác (data, output, sampling, training, lora) giữ đúng convention chuẩn
    (seed=42, sampling_mode=first_n, max_eval_examples=100). Đường dẫn adapter lấy từ
    config, KHÔNG hardcode trong script.

PHASE S02-2 — src/S02_minimal_coupling.py
  - class LinearProjectionBridge(nn.Module): forward(H_A) trong đó H_A shape
    (batch, seq_len, d_model) -> mean pool qua dim=1 -> nn.Linear(d_model,
    bottleneck_dim) -> trả về z shape (batch, bottleneck_dim). d_model lấy từ
    model.config.hidden_size, KHÔNG hardcode.
  - class GatedInjection(nn.Module):
      up_proj: nn.Linear(bottleneck_dim, d_model)
      gate: nn.Linear(d_model + d_model, 1)   # input là concat([h_B, z_proj])
      forward(h_B, z) -> z_proj = up_proj(z); gate_val = sigmoid(gate(cat([h_B,
      z_proj.expand_as(h_B)], dim=-1))); return h_B + gate_val * z_proj.expand_as(h_B)
  - class MinimalCouplingBridge(nn.Module): gộp cả 2 class trên, có save/load
    state_dict riêng (KHÔNG qua PEFT vì đây không phải LoRA, là module riêng).
  - Toàn bộ class có type hints đầy đủ, docstring ngắn.

PHASE S02-3 — scripts/S02_test_forward_single_example.py
  - Load Agent A (frozen, adapter từ agent_a_adapter_path trong config), Agent B
    (LoRA, init theo config từ agent_b_adapter_path).
  - Gắn HiddenStateExtractor lên layer_index của A, HiddenStateInjector lên
    layer_index của B với injection_fn dùng MinimalCouplingBridge.
  - Chạy forward trên đúng 1 mẫu GSM8K, in ra: shape của H_A, shape của z, shape sau
    injection, giá trị loss nếu tính thử cross-entropy trên 1 token, và xác nhận
    `bridge.parameters()` có `.grad` khác None sau `loss.backward()`.
  - Đây LÀ smoke test bắt buộc theo PHẦN 2 mục 1 — phải chạy thật trên GPU trước khi
    sang PHASE S02-4.

PHASE S02-4 — scripts/S02_train_minimal_coupling.py
  - Load Agent A: base + adapter (từ agent_a_adapter_path), set `requires_grad_(False)`
    toàn bộ (đóng băng hoàn toàn theo config freeze_agent_a).
  - Load Agent B: base + LoRA (init từ agent_b_adapter_path nếu init_agent_b_from_d11=
    true, dùng peft merge_and_unload rồi re-wrap LoRA mới, hoặc load thẳng adapter làm
    điểm khởi đầu rồi tiếp tục train — chọn cách nào đơn giản/ổn định hơn, ghi rõ lý
    do trong comment).
  - Khởi tạo MinimalCouplingBridge (S02-2), đưa params vào optimizer cùng với LoRA
    params của Agent B (KHÔNG đưa params của Agent A vào optimizer).
  - Training loop: với mỗi batch, forward A dưới `torch.no_grad()` để lấy H_A (không
    cần giữ graph vì A bị đóng băng — tiết kiệm memory đáng kể), sau đó forward B CÓ
    injection (H_A không cần grad nhưng bridge/injection layers thì cần), tính loss
    SFT cross-entropy trên output text của B so với gold reasoning+answer (dùng
    format_messages_with_assistant() từ src/prompts.py), backward, step.
  - Dùng argparse cho `--max-examples` (mặc định None = full), `--max-steps` (mặc
    định None = full theo epoch).
  - Data guards: reject_test_split_for_training(), reject_test_rows_for_training().
  - Save: LoRA của Agent B vào outputs/S02_minimal_coupling_gsm8k/adapters/
    agent_B_minimal_coupling_sft/, bridge state_dict vào outputs/
    S02_minimal_coupling_gsm8k/adapters/minimal_coupling_bridge.pt.
  - SMOKE TEST BẮT BUỘC: chạy với `--max-examples 2 --max-steps 4`, xác nhận loss in
    ra hợp lý (không NaN, giảm dần qua các step), rồi mới báo cáo lệnh full run.

PHASE S02-5 — scripts/S02_evaluate_minimal_coupling.py
  - 3 eval modes trên data/raw/test.jsonl (first_n=100, seed=42):
      1. agent_A_alone: Agent A (adapter import) giải full problem độc lập (KHÔNG qua
         bridge — đây chỉ để verify lại số 0.58 của D11.0, có thể tái dùng kết quả cũ
         nếu đã có, nhưng vẫn nên chạy lại trên đúng adapter đang dùng để chắc chắn).
      2. agent_B_alone: Agent B (adapter import GỐC, chưa qua train S02) giải full
         problem độc lập, không injection — baseline B trước khi train coupling.
      3. minimal_coupling: full pipeline A (frozen, forward lấy H_A) -> bridge -> B
         (LoRA đã train ở S02-4, CÓ injection) -> final answer.
  - Metrics JSON:
      {
        "experiment_name": "S02_minimal_coupling_gsm8k",
        "num_examples": 100,
        "agent_A_alone_accuracy": ...,
        "agent_B_alone_accuracy": ...,
        "minimal_coupling_accuracy": ...,
        "delta_vs_max_alone": minimal_coupling_accuracy - max(A_alone, B_alone),
        "delta_vs_d11_0_text_pipeline": minimal_coupling_accuracy - 0.68
      }
  - Lưu predictions JSONL đầy đủ (problem, gold, mode, prediction, correct) vào
    outputs/S02_minimal_coupling_gsm8k/generations/eval_predictions.jsonl.
  - Lưu metrics vào outputs/S02_minimal_coupling_gsm8k/metrics/eval_metrics.json.
  - SMOKE TEST BẮT BUỘC: chạy với `--max-examples 2` trước khi báo full run.

================================================================
PHẦN 5 — TIÊU CHUẨN HOÀN THÀNH & BÁO CÁO CHO USER
================================================================

1. Mọi script (S01-1 đến S02-5) đã chạy smoke test PASS thật trên GPU — không phải
   suy đoán, phải có output terminal thật làm bằng chứng.

2. In ra terminal (cuối cùng, sau khi mọi smoke test pass) danh sách đầy đủ các lệnh
   Bash để tôi tự chạy full run theo đúng thứ tự: verify adapter -> (nếu cần) retrain
   -> train minimal coupling full -> evaluate minimal coupling full 100 mẫu.

3. BẮT BUỘC viết notes/S01_S02_ssb_prototype_report.md — báo cáo đầy đủ, có cấu trúc
   tương tự D12_0_voting_report.md (repo cũ, chỉ đọc tham chiếu format, không copy nội
   dung), gồm các phần:
     a. Đặt vấn đề & Hướng đi: vì sao làm S01/S02 (SSB tối giản trên GSM8K) trước khi
        làm dataset mới hay Mamba đầy đủ, và vì sao tách sang repo mới cospec-ssb thay
        vì làm tiếp trong repo D11.x/D12.x cũ.
     b. Phương pháp: mô tả kiến trúc Minimal Coupling (Write Encoder tuyến tính, Gated
        Injection), có thể vẽ sơ đồ ASCII đơn giản A -> z -> B.
     c. Quyết định triển khai cụ thể: layer_index dùng, bottleneck_dim, vì sao đóng
        băng Agent A, cách khởi tạo Agent B (từ adapter import hay fresh) và lý do
        chọn, và trạng thái adapter import (còn nguyên hay phải retrain do lỗi file).
     d. Kết quả: bảng so sánh agent_A_alone / agent_B_alone / minimal_coupling /
        D11.0 text pipeline (0.68) / D12.0 majority vote (0.50), kèm delta.
     e. Đọc kết quả ban đầu: nêu rõ đây CHƯA phải kết luận cuối (chưa chạy causal
        test S03 — shuffle/zero/noise), chỉ là tín hiệu sơ bộ liệu pipeline có "sống"
        được không.
     f. Định hướng bước tiếp theo: dẫn sang S03 (causal diagnostic).

4. Đồng thời viết notes/S02_minimal_coupling_results.md bản ngắn đúng format chuẩn
   (tham khảo format D11_0_baseline_results.md, D12_0_voting_results.md ở repo cũ) để
   dễ tra cứu nhanh số liệu.

5. Không dùng framing solver/verifier cho Agent A/B ở bất kỳ đâu trong code, comment,
   hay report — giữ đúng triết lý trung tính của toàn bộ dự án.
````
