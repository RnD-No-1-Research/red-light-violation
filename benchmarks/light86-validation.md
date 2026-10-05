# Cam_04 bỏ sót ID 86: mất trạng thái đèn lúc cắt vạch

Ngày 30/09/2026, nguồn vid_4.mp4, vehicle_n_best.pt, ByteTrack hiện tại.
Đối chiếu lịch sử tracking và chạy lại model đèn trên toàn bộ 361 frame nguồn.
Chỉ số frame dưới đây bắt đầu từ 0.

| Frame | Giây video | Điểm dưới bbox ID 86 | Khoảng cách có dấu tới vạch, pixel | Đèn trước sửa |
|---|---:|---|---:|---|
| 201 | 33.443 | (226.5, 377) | +7.62 | RED |
| 202 | 33.610 | (227.5, 371) | +1.61 | RED |
| 203 | 33.776 | (228.5, 366) | -3.40 | UNKNOWN |

ID không bị mất hoặc đổi tại thời điểm cắt. Điểm chuyển từ phía dương sang âm,
cắt đúng đoạn vạch trong polygon. Model đèn trả UNKNOWN ở ba frame 201–203;
bộ làm mượt 5 frame đổi sang UNKNOWN ở 203–205. Điều kiện `light == "RED"`
không đạt tại frame 203. Khi RED trở lại ở 206, xe đã ở sau vạch, không còn
sự kiện đổi phía để ghi. Tính năng nghi vấn che khuất không áp dụng vì xe vẫn
được quan sát liên tục.

## Hiệu chỉnh

Giữ ROI chính `[614, 178, 26, 77]` và ngưỡng đèn 0.15. Riêng cam_04 thêm:

```yaml
traffic_light_fallback_roi: [614, 200, 26, 55]
traffic_light_canvas_size: 256
```

Khi ROI chính trả UNKNOWN, chạy lại model trên vùng thân đèn (bỏ bảng đếm ngược),
đặt giữa ảnh đệm đen 256×256. Cơ chế này đã có trong TrafficLightDetector.
Không giữ màu cũ, không đổi UNKNOWN thành RED bằng quy tắc, không thay ByteTrack
hoặc điều kiện hình học. Màu hợp lệ ở ROI chính không bị ghi đè.

Đã kiểm tra tuần tự cả 361 frame: 24 lần ROI chính UNKNOWN, dự phòng nhận lại
23 lần (6 RED, 17 GREEN), còn 1 UNKNOWN. Sau smoothing, các frame duy nhất
được bổ sung RED so với trước sửa là 203–205, đúng khoảng cần khôi phục.
Pha GREEN/YELLOW/RED khác vẫn có thể bị model chính nhận sai; đây không phải
đo accuracy trên tập nhãn độc lập. Cấu hình phụ thuộc vị trí camera, cần vẽ lại
khi đổi góc/kích thước video.

Cách đệm toàn ROI 192 px đã thử nhưng không chọn vì có thể nhận sai màu ở pha
xanh. Số liệu theo frame của cấu hình cuối: [light86-frames.json](light86-frames.json).
Kết quả chạy pipeline thật và đường dẫn video/bằng chứng:
[light86-validation.json](light86-validation.json).

Runtime kiểm chứng: Python 3.12, torch CPU 2 threads, Ultralytics 8.4.165 nạp
từ source venv project. Chưa kiểm thử CUDA/Python 3.13 hoặc RTSP thật.
Chi phí bổ sung: tối đa một lần suy luận đèn khi ROI chính UNKNOWN; trên clip
này phát sinh ở 24/361 frame. Không chạy lại model xe.
