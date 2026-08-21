# Lab 21 — Evaluation Report

**Họ tên**: Lê Nguyễn Phi Trường  **MSSV**: 2A202601541  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB VRAM)`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu huấn luyện) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 steps |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào? Template giữ nguyên vẹn khối `<think>` và thẻ đóng `</think>`, đảm bảo các reasoning traces không bị strip trong quá trình tokenize/render.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.758 | 0.000 | 3230.3 |
| (b) base + optimized prompt | 0.765 | 0.758 | 1.000 | 1058.3 |
| (c) LoRA fine-tune | 0.970 | 0.611 | 1.000 | 1413.9 |

**(b) có thật sự mạnh hơn (a) không?** Có — (b) đạt 0.765 target so với 0.000 của (a), định dạng JSON đúng 100% (1.000) và độ trễ giảm xuống còn 1058.3 ms.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao? Không sửa, giữ nguyên prompt tối ưu gốc được chuẩn hóa với SHA `719e74d3b6232053` để đảm bảo tính liêm chính của phép so sánh đối chứng.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6265 | 0.970 | 936.5 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5365 | 0.970 | 807.7 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.000 | 948.2 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7070 | 0.940 | 1013.5 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt độ chính xác 0.970, tức là hòa điểm tuyệt đối với `correct` (0.970). Tuy nhiên, trên thang đo train loss, `attn_only` lại có loss thấp hơn (0.5365 so với 0.6265 của `correct`). Sự bất đồng thứ tự giữa train loss và target accuracy chứng minh rằng train loss thấp hơn có thể chỉ do overfit/ghi nhớ dữ liệu huấn luyện cục bộ trên 2 ma trận Attention khi rank được đẩy lên rất cao ($r=283$). Kết quả này khẳng định: vị trí gắn adapter phân bổ đều toàn bộ mạng (`text-linear`) mang lại tính ổn định biểu diễn tốt hơn, trong khi việc cố tình tăng rank để bù đắp cho việc thiếu vị trí gắn chỉ tạo ra ảo giác giảm loss trong tập train mà không cải thiện thêm năng lực thực tế trên tập kiểm thử.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Đường train loss của `wrong_lr` dừng lại ở mức 1.5704 sau 30 step (cao gấp hơn 2.5 lần so với `correct` 0.6265), khiến target accuracy và format hoàn toàn bị tê liệt ở mức 0.000. Nếu chỉ nhìn vào loss mà không biết learning rate đang bị đặt ở thang Full-FT ($10^{-5}$), người kỹ sư rất dễ kết luận sai rằng "phương pháp LoRA không hoạt động trên bài toán này" hoặc "dataset quá khó khiến mô hình không thể hội tụ". Thực tế, bản chất lỗi chỉ nằm ở việc bước cập nhật gradient quá nhỏ đối với ma trận adapter khởi tạo ngẫu nhiên, khiến trọng số chưa kịp dịch chuyển khỏi điểm xuất phát.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
Run `qlora` giúp tiết kiệm được 4.92 GB VRAM (chỉ sử dụng 7.09 GB so với 12.01 GB của bản 16-bit). Tuy nhiên, cái giá phải trả là thời gian huấn luyện lâu hơn (1013.5 giây so với 936.5 giây), độ trễ suy luận latency tăng 26% (1781.1 ms so với 1413.9 ms), và độ chính xác target bị suy giảm nhẹ từ 0.970 xuống còn 0.940 do sai số lượng tử hóa 4-bit NF4. Số liệu thực nghiệm hoàn toàn ủng hộ khuyến nghị kỹ thuật: đối với dòng kiến trúc Qwen3.5 trên phần cứng có đủ VRAM (như T4 16GB), nên ưu tiên dùng 16-bit LoRA chuẩn để đạt độ chính xác tối đa và tốc độ suy luận nhanh hơn, chỉ dùng QLoRA khi bị ép buộc bởi giới hạn bộ nhớ dưới 8GB.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.205` · `regression Δ = -0.147` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)
Kết quả cổng hồi quy trả về **FAILED** do độ chính xác trên bộ kiểm thử hồi quy tổng quát (`eval_regression.jsonl`) bị sụt giảm nghiêm trọng từ 0.758 xuống 0.611 (tụt $\Delta = -0.147$, vượt quá ngưỡng dung sai cho phép là 0.020). Trong khi đó, ở tác vụ đích CSKH, bản Fine-tune LoRA đạt mức cải thiện rất ấn tượng $\Delta = +0.205$ (tăng từ 0.765 lên 0.970). 
Hiện tượng này phản ánh đúng vấn đề kinh điển trong kỹ nghệ AI: **Hiện tượng quên thảm họa (Catastrophic Forgetting)**. Khi ta huấn luyện mô hình 30 steps chỉ thuần túy trên 225 mẫu ticket CSKH đặc thù, các trọng số LoRA bị chuyên biệt hóa quá mức vào cấu trúc JSON của bài toán hẹp, làm suy thoái năng lực suy luận và trả lời câu hỏi kiến thức phổ thông đã học trong pha tiền huấn luyện. Về mặt kỹ thuật, kết quả FAILED này chỉ ra rằng bài toán phân loại CSKH này chưa an toàn để triển khai đơn lẻ nếu không áp dụng giải pháp bù đắp: cần trộn thêm 1–5% dữ liệu đệm đa năng (replay data) vào tập huấn luyện để bảo vệ năng lực tổng quát của mô hình nền.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper không giao... | `{"intent": "van_chuyen", "urgency": "thap", ...}` | `{"intent": "hoi_thong_tin", ...}` | `{"intent": "van_chuyen", "urgency": "thap", ...}` | ✅ FT thắng: Nhận diện chính xác intent vận chuyển thay vì nhầm sang hỏi thông tin như prompt. |
| 2 | Chào shop, mình đặt ốp lưng điện thoại mã đơn VN833689. Sai màu. Sớm nhé... | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | `{"intent": "doi_tra", ...}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ✅ FT thắng: Phân biệt đúng lỗi sản phẩm (sai màu) với yêu cầu đổi trả thông thường. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện... | `{"intent": "hoan_tien", "urgency": "thap", ...}` | `{"intent": "hoan_tien", "urgency": "thap", ...}` | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Model FT bị bias cụm từ "Chưa thấy tiền" nên nâng mức urgency lên `trung_binh`, trong khi nhãn đúng là `thap` do có câu "Khi nào tiện". |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện... | `{"intent": "san_pham_loi", "urgency": "thap", ...}` | `{"intent": "san_pham_loi", "urgency": "thap", ...}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Tương tự ca 3, mô hình FT gán nhãn urgency quá mức nhạy cảm (`trung_binh` thay vì `thap`), prompt gốc xử lý chính xác hơn. |
| 5 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop... | `{"intent": "san_pham_loi", "urgency": "thap", ...}` | `{"intent": "san_pham_loi", "urgency": "thap", ...}` | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Khi xuất hiện từ khóa "Bị lỗi", LoRA FT có xu hướng mặc định urgency tối thiểu là `trung_binh`, bỏ qua ngữ cảnh giảm nhẹ "Khi nào tiện". |

Có mẫu chung nào ở các ca FT thua không?
Mẫu chung rõ rệt nhất là: **Mô hình Fine-tune bị thiên kiến (bias) quá mức về mức độ khẩn cấp (Urgency)**. Cụ thể, trong tập dữ liệu huấn luyện, các ticket chứa từ khóa tiêu cực hoặc báo lỗi thường gắn liền với mức urgency từ `trung_binh` đến `cao`. Do đó, khi gặp các ticket vừa có lỗi vừa có câu giảm nhẹ như "Khi nào tiện / Không vội", mô hình FT ưu tiên bắt từ khóa lỗi và dự đoán sai thành `trung_binh`, trong khi Base Model với Optimized Prompt lại giữ được khả năng suy luận logic cân bằng hơn để gán đúng nhãn `thap`.

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** 
Dựa trên kết quả thực nghiệm khách quan từ pipeline đo đạc 4 nhóm tiêu chí, tôi đưa ra kết luận rằng: **Chưa nên triển khai trực tiếp bản LoRA Fine-tune này vào môi trường Production mà cần tiếp tục cải tiến dữ liệu**. 
Mặc dù bản Fine-tune mang lại bước nhảy vọt về độ chính xác trên nghiệp vụ CSKH (đạt 97.0% so với 76.5% của Optimized Prompt, format JSON chuẩn 100%), việc mô hình bị trượt bài kiểm tra hồi quy tổng quát ($\Delta = -0.147$) chứng tỏ trọng số adapter đã gây tổn hại đến tri thức nền tảng của Base Model. 
Qua toàn bộ thí nghiệm A/B Testing, tôi nhận thấy **đòn bẩy thực sự quyết định sự thành bại của pipeline không phải là rank LoRA hay kiến trúc phần cứng, mà là Loss Masking và Learning Rate**. Nếu Loss Mask sai (tính loss trên cả câu hỏi), mô hình sẽ học vẹt; nếu Learning Rate sai thang độ ($10^{-5}$ thay vì $10^{-4}$), mô hình hoàn toàn bất động. Đứng ở góc độ kỹ sư phần mềm, một giải pháp AI chỉ thực sự sẵn sàng khi nó vượt qua cả Integration Test (Target) lẫn Regression Test (Safety) một cách an toàn.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Loss Masking là Unit Test bắt buộc của SFT**: Không bao giờ được tin tưởng dữ liệu huấn luyện nếu chưa giải mã ngược (decode) mảng nhãn `labels` để assert rằng token câu hỏi mang giá trị `-100` và chỉ có câu trả lời của Bot được tính gradient.
2. **Train Loss là chỉ số thay thế nguy hiểm**: `attn_only` có train loss thấp hơn `correct` (0.5365 vs 0.6265) nhưng độ chính xác target thực tế trên tập kiểm thử chỉ bằng nhau (0.970). Đánh giá mô hình phải dựa trên năng lực downstream task thực tế, không dựa vào loss.
3. **Prompt Engineering là Baseline bắt buộc phải vượt**: Không bao giờ so sánh Fine-tuning với Naive Prompt (0.0%). Phải luôn thiết lập một Optimized Prompt mạnh mẽ (76.5%) làm mốc chuẩn để đánh giá xem chi phí huấn luyện và bảo trì adapter có thực sự mang lại ROI (Return on Investment) xứng đáng hay không.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Trộn 3–5% dữ liệu đa năng (Replay Data từ tập UltraChat/OpenOrca tiếng Việt) vào tập dữ liệu huấn luyện 225 mẫu CSKH, sau đó train lại với `correct` configuration để khắc phục triệt để hiện tượng Quên thảm họa và đưa phán quyết cổng hồi quy từ `FAILED` thành `PASSED`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
