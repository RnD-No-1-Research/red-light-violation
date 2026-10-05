# Xe đội mũ xanh xuất hiện lần đầu sau vạch — cam_03

Kiểm chứng 01/10/2026, vid_3.mp4, 640×480, 384 frame, vehicle_n_best.pt, ngưỡng
xe 0.5. ID trong lượt chạy riêng là 61; khi chạy nhiều camera ID có thể khác.

## Quan sát và kết quả

- Frame 102 (từ 0), 16.953s: lần đầu tracker trả ID 61,
  bbox `[460,249,496,328]`. Điểm dưới bbox đã cách vạch 30.555 pixel về phía sau.
- Xe gần như dừng tại chỗ; không có cặp quan sát trước/sau vạch, cũng không có
  lịch sử cùng ID để áp dụng nhánh nghi vấn mất dấu.
- Quy tắc mới theo yêu cầu người dùng: sau 3 quan sát liên tiếp, frame 104,
  **17.286s**, bbox `[460,252,496,327]`, ghi vi phạm và giữ bbox đỏ.
- Metadata ghi `LATE_DETECTION_AFTER_LINE`, `observed_crossing: false`,
  `motion_basis: STATIONARY_AFTER_LINE`; lưu đủ thời gian, vị trí xuất hiện đầu,
  số frame xác nhận. Không trình bày đây là một lần cắt vạch được quan sát trực tiếp.

Chạy pipeline thật toàn bộ 384 frame, kiểm tra dòng ID 61 trong violations.csv,
metadata đúng lý do, JPEG đọc được; MP4 đủ 384 frame và đọc được frame cuối.
Đường dẫn output và số liệu: [late-detection-validation.json](late-detection-validation.json).

## Hiệu chỉnh đèn cam_03

ROI chính `[609,144,29,75]` trả UNKNOWN ở nhiều frame, bao gồm lúc xe xuất hiện.
Giữ ROI chính, thêm dự phòng `[612,170,24,49]` trên canvas 256. Chỉ chạy dự phòng
khi lượt chính UNKNOWN, cùng ngưỡng 0.15, không giữ màu đỏ cũ.

Đã kiểm tra tuần tự cả clip, bao gồm pha xanh. Sau smoothing cấu hình cuối:
RED từ frame 4 tới 221; GREEN bắt đầu 222; một số khoảng sau đó còn UNKNOWN.
Không phát sinh RED sau frame 221 trong kết quả làm mượt của clip kiểm tra này.
Đây là kiểm chứng trên clip hiệu chỉnh, chưa có tập nhãn độc lập để công bố
accuracy. [Màu đèn từng frame](late-detection-light-frames.json).

## Kiểm thử và giới hạn

131 test qua: bật/tắt theo camera, pha đèn và khởi động/reconnect, khoảng mất
detection, vùng sau vạch, xe ngang/ngược chiều, biến dạng bbox, xe đứng yên,
ca bbox thật của ID 61, chống ghi trùng, CSV/JSON và bbox đỏ trong controller.

Runtime: Python 3.12, torch CPU 2 threads, source Ultralytics 8.4.165 nạp từ venv
project. Chưa xác minh Python 3.13/CUDA/RTSP thật. Thời gian trong JSON chỉ là
lượt kiểm chứng chức năng, không dùng làm benchmark so sánh trước/sau.

Nhánh xe mới không chạy thêm model xe; nhận diện đèn dự phòng tốn thêm một lần
inference đèn khi ROI chính UNKNOWN. Ghi vi phạm mới cũng có thể gọi plate/OCR.

Giới hạn nghiệp vụ: xe ở sau vạch có thể đã vào từ pha xanh, từ hướng khác hoặc
được gán ID mới. Các bộ lọc giảm một số tình huống, không chứng minh thời điểm
xe qua vạch. Lượt kiểm chứng này ban đầu bật riêng cam_03. Sau đó đã chuyển sang
mặc định chung cho mọi camera theo yêu cầu người dùng; xem
[kiểm chứng đa camera](all-camera-late-validation.md).
