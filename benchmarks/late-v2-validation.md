# Kiểm chứng phát hiện muộn — 01/10/2026

## Vấn đề và thay đổi

Người dùng chỉ rõ cam_01 bỏ sót ở 10 giây đầu và 5 giây cuối. Video chỉ khoảng
3,016 FPS. Quy tắc cũ loại ID vĩnh viễn nếu lần đầu thấy khi đèn chưa đủ thời gian
ổn định, đòi 3 frame nguồn liên tiếp trong 1 giây và tiến theo pháp tuyến ít nhất
một nửa độ dịch dọc theo vạch. Những điều kiện này làm mất xe đi nhanh, mất dấu
ngắn hoặc đi chéo.

Bản sửa dùng chung cho mọi camera:

- Giữ quyền xét lại cho ID mới xuất hiện khi đèn thô đỏ và ID đã thấy trước vạch;
  không dùng việc “ID từng xuất hiện” để loại vĩnh viễn.
- Tính thời gian đỏ từ kết quả thô trong lúc bộ làm mượt khởi động, nhưng chỉ thu
  thập ứng viên khi cả màu thô và màu làm mượt đều đỏ.
- Chuỗi tối đa 2 giây; khoảng gián đoạn tối đa 0,75 giây. Hết hạn thu thập lại
  chuỗi mới nếu đủ điều kiện. Xanh/vàng xóa quyền xét lại cũ.
- Chuyển động tiến rõ cần 2 quan sát; đứng yên vẫn cần 3 quan sát.
  Tỷ lệ tiến pháp tuyến / dịch dọc theo vạch giảm từ 0,5 xuống 0,25.
- Lần cắt vạch bị bộ lọc biến dạng bbox từ chối không được nhánh muộn ghi bù.
- Theo yêu cầu người dùng, giữ `((x1+x2)/2, y2-20)` thống nhất cho các nhánh và
  chấm hiển thị. Polygon vẫn lọc bằng đáy bbox. Không chỉnh lại vạch/ROI của người dùng.

## Cách kiểm chứng

Python 3.12, PyTorch 2.8 CPU, mã Ultralytics 8.4.165 từ môi trường local;
`vehicle_n_best.pt`, ngưỡng xe 0,3, đèn 0,25, biển 0,8, imgsz 640.
Không dùng số liệu này để khẳng định đã kiểm thử Python 3.13/CUDA/RTSP thật.
Tọa độ vạch, ROI, polygon và cấu hình hiệu lực được lưu trong
[JSON kết quả](late-v2-validation.json).

Chạy pipeline model thật đồng thời 4 video, gồm OCR khi có sự kiện, ghi bằng
chứng/CSV/MP4. Cả 4 camera kết thúc; pipeline trả 0. Sau đó kiểm tra số frame và
giải mã frame cuối của từng video. Script tổng hợp ban đầu lỗi do trùng khóa
`track_id`; đã sửa bước tổng hợp và kiểm tra lại các file kết quả hiện có.

| Camera | Frame vào/ra | Sự kiện dự đoán | Trong đó phát hiện muộn |
|---|---:|---:|---:|
| cam_01 | 192/192 | 26 | 13 |
| cam_02 | 384/384 | 12 | 0 |
| cam_03 | 384/384 | 10 | 3 |
| cam_04 | 361/361 | 7 | 2 |

Tổng 1321 frame. Các ảnh bằng chứng tồn tại, metadata đọc được, không lặp track ID
trong CSV của từng phiên. Thư mục video:
`output/late_v2_20261001_132932/<camera_id>/videos/`.

## Đối chiếu cam_01 trước/sau

Chạy riêng cam_01 thêm một lượt, lưu đúng quan sát YOLO/ByteTrack/màu đèn.
Phát lại **cùng chuỗi quan sát, cùng vạch và cùng điểm y2-20**, chỉ thay thuật toán
phát hiện muộn cũ/mới. Kết quả mới khớp toàn bộ ID và frame sự kiện của controller
thật. Đã giải mã đủ 192 frame MP4 của lượt riêng.

| Khoảng video | Logic cũ | Logic mới |
|---|---:|---:|
| 0–10 giây | 10 | 14 |
| 5 giây cuối | 6 | 7 |
| Toàn video | 18 | 26 |

Các ID được ghi thêm trong lượt chạy riêng:

| ID | Thời điểm quyết định | Ghi chú |
|---|---:|---|
| 9 | 1,658s | Có lịch sử trước vạch, được xét lại khi đèn rõ |
| 12 | 1,658s | Không bị loại vì xuất hiện lúc đèn chưa đủ thời gian chờ |
| 13 | 4,311s | Lấy lại track sau vạch và đủ quan sát |
| 42 | 7,958s | Chuyển động rõ trong 2 quan sát |
| 191 | 43,437s | Được xét lại sau thời gian chờ đỏ |
| 198 | 54,048s | Thu thập 6 quan sát, khoảng 1,658 giây, xe đi chéo |
| 201 | 57,364s | Thu thập lại sau đèn không rõ, đủ điều kiện đứng yên |
| 213 | 63,332s | Đủ tiến theo hướng chéo ở cuối video |

ID có thể đổi giữa các lượt, đặc biệt khi nhiều camera chạy đồng thời; bảng này
chỉ áp dụng cho video cam_01 chạy riêng đã lưu.

- Đầu video 1,658s: [trước](late-v2-early-before.jpg) / [sau](late-v2-early-after.jpg).
- Cuối video 63,332s: [trước](late-v2-end-before.jpg) / [sau](late-v2-end-after.jpg).
- [Ô tô cam_04 vẫn xanh tại 36,771s](late-v2-car-check.jpg); ID 1 trong lượt đa camera
  không nằm trong danh sách vi phạm. Bộ lọc bbox vẫn hoạt động, nhưng một trường
  hợp kiểm tra không chứng minh đã loại hết mọi vi phạm giả do bbox.

Video cam_01 chạy riêng:
`output/late_v2_cam01_20261001_133242/cam_01/videos/cam_01_20261001_133245_52f038806297_001.mp4`.

## Kiểm thử và giới hạn

153 pytest đạt; Ruff đạt; `main.py --check-config` nhận đủ 4 camera hợp lệ.
Test hồi quy bao gồm xe đi chéo FPS thấp, chờ hơn 1 giây, gap ngắn/dài, khởi động,
đổi pha khi xe mất dấu, bbox bị từ chối, cùng ID ở nhiều camera, hai chiều vạch,
vạch dọc và vạch chéo. Chạy lại từ thư mục project:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
.\venv\Scripts\python.exe main.py --camera_id cam_01
```

Đây là kiểm chứng hành vi và đầu ra, **không phải precision/recall hay số vi phạm
đúng**. Chưa có ground truth sự kiện; cần xem và gán nhãn trước khi kết luận accuracy.
Xe chỉ xuất hiện một quan sát, xe sẵn sau vạch ở frame đầu, ngoài polygon hoặc
không có detection vẫn có thể bị bỏ sót. Nhánh muộn là suy luận hiện diện/chuyển
động, không chứng minh thời điểm cắt vạch; metadata giữ `observed_crossing: false`.

ROI cam_01 trong video chứa đèn mũi tên phải màu đỏ, trong khi có đèn xanh khác
trong cảnh. Pipeline hiện chỉ nhận màu, chưa gán quyền đi thẳng/rẽ theo từng làn.
Vì vậy không được xem mọi bbox đỏ là một vi phạm giao thông đã được xác minh.

Không thêm lượt YOLO phương tiện hay bộ đệm ảnh cho logic mới. Nhiều sự kiện hơn
có thể tăng số lượt detect biển/OCR/ghi ảnh; chưa đo benchmark FPS so sánh của bản này.
