# Cam_04: bbox ô tô co giãn tạo vi phạm giả

Ngày 30/09/2026. Model vehicle_n_best.pt, ngưỡng 0.5, imgsz 640, ByteTrack giữ
tham số hiện tại; cấu hình đèn dự phòng cam_04 sau sửa ID 86.

## Quan sát

Ô tô ID 3 giữ lớp `car`, không đổi ID tại thời điểm gây sự kiện. Bounding box
YOLO thô đã thay đổi đáy; không phải riêng ByteTrack gộp hai track:

| Frame (từ 0) | Giây | Bbox sau tracker |
|---|---:|---|
| 200 | 33.277 | [515, 270, 610, 360] |
| 217 | 36.105 | [508, 270, 615, 380] |
| 220 | 36.604 | [510, 270, 612, 376] |
| 221 | 36.771 | [515, 270, 609, 367] |

Mép trên y=270 gần như cố định, đáy bị kéo xuống rồi co lên qua vạch. Tại 221,
bbox YOLO thô cũng có đáy 362.497, so với 375.185 ở frame 220. Điều kiện cũ chỉ
xét điểm giữa đáy, vì vậy đã ghi `car #3 VI PHAM`.

## Thay đổi

`box_motion_issue()` kiểm tra cùng lớp (ở caller), bbox hợp lệ, tỷ lệ rộng/cao
trong [0.5, 2]. Tính dịch chuyển hai trung điểm mép trên/dưới theo pháp tuyến của
vạch, dấu dương theo chiều vi phạm. Nếu đáy di chuyển qua vạch mà:

```text
top_motion + 1 pixel < 0.25 * bottom_motion
```

thì từ chối lần cắt với lý do `box_deformation`. Sai số 1 pixel cho phép tọa độ
tracker làm tròn; không phải ngưỡng tốc độ vật lý. Áp dụng cả vi phạm trực tiếp
và nhánh nghi vấn mất dấu. Lần cắt bị từ chối không thêm ID vào tập vi phạm,
không OCR/ghi evidence vi phạm, có log lý do; vị trí mới vẫn được cập nhật để
không nối lại qua lịch sử cũ. Không thêm inference hoặc bộ đệm ảnh.

## Kiểm chứng và giới hạn

111 test qua: hồi quy 8 cặp bbox cắt vạch đã quan sát trong cam_04, đổi lớp,
nhảy kích thước, hai chiều/đường chéo, chạm vạch và nhánh nghi vấn không được
tạo lại sự kiện cho bbox ô tô biến dạng. Giữ phép xét điểm không có metadata
bbox cho các lời gọi geometry độc lập cũ; controller luôn truyền metadata.

Chạy lại model thật toàn bộ 361 frame: kiểm tra ID 3 không xuất hiện trong
violations.csv hoặc review_events.csv; ID 86 vẫn ghi vi phạm, ID 82 vẫn là nghi
vấn. Các sự kiện khác của lượt trước giữ nguyên ID: 56,84,85,86,83,91,113.
Đây chỉ là đối chiếu không làm mất sự kiện đã có, không xác nhận mọi sự kiện
đó là đúng theo ground truth. Kết quả: [boxmotion-validation.json](boxmotion-validation.json).

[Ảnh output tại frame 221 sau sửa](boxmotion-after.jpg): ô tô ID 3 giữ bbox xanh.

Giới hạn: đây là bộ lọc biến dạng rõ ràng, không tách hai xe trong bbox, không
chứng minh danh tính, không xử lý mọi ID switch hoặc jitter nhỏ. Xe di chuyển
thật nhưng bị che khuất làm mép trên đứng yên có thể bị bỏ sót. Cần dữ liệu gán
nhãn nhiều tình huống để đánh giá precision/recall và chọn ngưỡng tổng quát.
Kiểm chứng dùng Python 3.12, torch CPU 2 threads, Ultralytics 8.4.165 source;
chưa xác minh CUDA/Python 3.13/RTSP thật.
