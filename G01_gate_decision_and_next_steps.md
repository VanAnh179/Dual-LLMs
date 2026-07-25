# G01 — Gate Decision & Định Hướng Tiếp Theo (SSB Fast Track)

**Ngày quyết định:** dựa trên kết quả S01/S02/S03 (GSM8K track) và V01 (dataset track) đã hoàn thành.

---

## 1. Đầu Vào Cho Quyết Định

### Track S — SSB trên GSM8K

| Nguồn | Số liệu |
|---|---|
| S02: minimal_coupling | **0.37** (thấp hơn cả B alone 0.61 lẫn D11.0 text pipeline 0.68) |
| S03 verdict | **NO_CAUSAL_DEPENDENCY**, `s04_gate: HOLD` |
| Δ_shuffle | 0.00, CI [-0.03, 0.03] — hoán vị z không ảnh hưởng gì |
| Δ_zero | -0.01, CI [-0.06, 0.04] — set z=0 không ảnh hưởng gì |
| matched vs zero-trained control | -0.06, CI [-0.14, 0.02] — matched còn *thấp hơn* control (dù CI chứa 0) |
| **zero-trained control vs B alone** | **0.43 vs 0.61 — thấp hơn 18 điểm, dù z đã bị ép = 0** |

### Track V — Dataset V01

| Nguồn | Số liệu |
|---|---|
| Leakage probe (tuyến tính) | view_a = 0.25, view_b = 0.25 — đúng bằng random (threshold 0.28) |
| **Neural baseline (LLM thật, zero-shot)** | view_a = **0.25**, view_b = **0.25** (trên phần parse được), full_problem = **0.89** |
| Gate | `PASS` |

---

## 2. Phân Tích: Vì Sao S03 Cho Kết Quả Âm Tính — Hai Giả Thuyết Cạnh Tranh

**Giả thuyết 1 — Ceiling effect của GSM8K.** B alone (retrained) đã đạt 0.61/1.00 — B tự giải được phần lớn bài toán mà không cần bất kỳ thông tin gì từ A. Không có áp lực nào buộc B phải học cách dùng z trong lúc training, nên mô hình có thể "học cách bỏ qua" injection một cách an toàn mà vẫn tối ưu được loss.

**Giả thuyết 2 — Confound từ chính quy trình training.** Ngay cả ở nhánh **zero-trained control** (kiến trúc giống hệt, chỉ z=0), B vẫn tụt 18 điểm so với B alone gốc. Điều này gợi ý một phần kết quả xấu không đến từ "z vô dụng" mà từ chính cách retrain hiện tại (merge_and_unload + fresh LoRA trên chỉ 180 mẫu, ~23 step) đang **làm hỏng năng lực gốc của B**, độc lập hoàn toàn với injection.

Hai giả thuyết này **chưa được tách bạch** — và nếu leo thẳng lên Mamba (S04) trong khi cả hai vẫn còn treo, bất kỳ kết quả nào từ Mamba cũng sẽ bị confound tương tự, không diễn giải được.

---

## 3. Quyết Định

### ❌ KHÔNG đi thẳng lên S04 (nâng cấp Mamba)

Lý do: leo thang kiến trúc trong khi chưa biết (a) dataset có thực sự ép buộc cộng tác hay không, và (b) quy trình training có đang tự gây hại hay không — là đầu tư sai chỗ.

### ✅ Bước tiếp theo: S05 — Retry Minimal Coupling (vẫn tuyến tính, CHƯA Mamba) trên V01, với 1 thay đổi thiết kế mấu chốt: **đóng băng CẢ Agent A lẫn Agent B hoàn toàn (không LoRA cho cả hai), CHỈ train Bridge**

Thiết kế mới loại bỏ được **cả hai giả thuyết cạnh tranh cùng lúc, trong 1 thực nghiệm duy nhất**:

- **Loại giả thuyết 1 (ceiling effect):** V01 neural baseline (LLM thật, không phải linear probe) đã chứng minh B alone chỉ đạt ~0.25 — đúng bằng random. B **thực sự** cần thông tin từ A để vượt qua mức này. Áp lực buộc dùng z là có thật, không còn ceiling effect.
- **Loại giả thuyết 2 (training regime confound):** Vì Agent B **không train thêm bất kỳ tham số nào**, không còn "vòng training thêm" nào có thể làm hỏng năng lực gốc của B. Optimizer chỉ chạm vào Bridge (Write Encoder + Gate — vài trăm nghìn tham số), không đụng gì đến LoRA hay base weights.

Đây là thiết kế thực nghiệm sạch nhất có thể để trả lời đúng 1 câu hỏi: **"z có mang thông tin hữu ích mà B cần hay không?"** — không còn confound nào khác xen vào phép đo.

**Lợi ích phụ:** vì B không cần train, quy mô training nhỏ hơn hẳn (chỉ Bridge, vài trăm nghìn tham số) — nhanh hơn S02 nhiều, và không cần lo lắng về catastrophic forgetting hay merge_and_unload nữa.

---

## 4. Rủi Ro Còn Lại Cần Theo Dõi

- Nếu S05 **vẫn** cho verdict NO_CAUSAL_DEPENDENCY sau khi đã loại bỏ cả 2 confound trên, đó là bằng chứng khá dứt khoát rằng chính cơ chế Minimal Coupling (mean-pool + linear projection + gated injection tại layer 14, bottleneck 64) không đủ biểu đạt — lúc đó redesign kiến trúc (Mamba, đổi layer_index, tăng bottleneck_dim) mới thực sự cần thiết, và biết chắc lý do là kiến trúc chứ không phải dataset hay training regime.
- `layer_index=14`, `bottleneck_dim=64` giữ nguyên từ S02 để so sánh trực tiếp được — nhưng đây vẫn là 2 giả định chưa qua ablation, cần quay lại nếu S05 cũng cho kết quả âm.
- V01 là bài toán phân loại 4 lớp (SLOT_0..SLOT_3), khác hẳn GSM8K (sinh số mở) — cần điều chỉnh prompt/loss/parse answer cho phù hợp, xem chi tiết trong prompt triển khai S05.

---

## 5. Gate Decision (machine-readable)

```yaml
s04_mamba_gate: HOLD
s05_gate: GO
dataset_track: V01
dataset_status: PASS
next_experiment: S05_bridge_only_minimal_coupling_v01
confounds_addressed_by_s05: [ceiling_effect, training_regime_degradation]
reserved_for_later: S04_mamba_upgrade  # chỉ triển khai nếu S05 cho tín hiệu causal thật
```
