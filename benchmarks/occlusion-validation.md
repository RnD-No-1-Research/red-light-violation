# Kiểm chứng nghi vấn vượt vạch khi bị che khuất — 30/09/2026

## Kết quả chức năng

Chạy model thật `vehicle_n_best.pt`, giữ ngưỡng xe 0.5, imgsz 640 và ByteTrack
hiện tại. Nguồn `vid_4.mp4`, cam_04, 181 frame đầu (0..180), cấu hình polygon,
vạch và đèn đang dùng. Tại frame 156, ID 82 còn trước vạch; thiếu detection riêng
ở 157..159; xuất hiện sau vạch tại 160. Khoảng hai quan sát: **0.665536 giây**.

- Tắt chức năng: không tạo sự kiện cho ID 82.
- Bật 1 giây: tạo đúng một `SUSPECTED_OCCLUDED_CROSSING` cho ID 82, ảnh trước tại
  25.955898s và ảnh sau tại 26.621434s. Bbox cam tiếp tục giữ khi thấy ID này.
- Cả bốn lượt đều có một vi phạm trực tiếp khác; ID 82 không vào violations.csv.
- Cả bốn MP4 đều đủ 181 frame, đọc được frame cuối. Hai ảnh mỗi nghi vấn đều
  giải mã được. Đã xem trực tiếp ảnh trước/sau của lượt kiểm chứng ban đầu.

Đây là kiểm chứng một lỗi cụ thể, chưa phải đo precision/recall hay bằng chứng
giải quyết mọi tình huống che khuất/đổi ID. Bộ lọc hình học chỉ giảm các trường
hợp gán nhầm rõ ràng; cùng ID vẫn có thể là xe khác.

## Chi phí thực đo

Môi trường: CPU Ryzen 5 4600H, torch 2.8.0+cpu, 2 threads, Python 3.12.14.
Dùng source Ultralytics 8.4.165 từ venv Python 3.13 của project, nạp vào runtime
3.12 để đo vì executable Python 3.13 không truy cập được từ phiên công cụ.
Không khẳng định các con số này là benchmark trên Python 3.13/CUDA của người dùng.

Model nạp một lần, warm-up xe/đèn trước đo; bốn lượt cùng process theo thứ tự
**tắt → bật → bật → tắt**. Tắt GUI, giữ ghi MP4, OCR khi có vi phạm trực tiếp,
JPEG/CSV bằng chứng. Reset counter tracker giữa lượt để đối chiếu cùng ID.
Lượt đo dưới đây không chạy pytest song song. Một lượt thử trước đó có chạy test
song song được loại khỏi bảng vì tải CPU không đồng nhất.

| Lượt | Chức năng | Tổng giây | FPS | Kiểm tra nghi vấn trung bình ms/frame | Ảnh lịch sử tối đa MiB |
|---|---|---:|---:|---:|---:|
| 1 | Tắt | 16.491 | 10.976 | 0.00363 | 0 |
| 2 | Bật | 14.617 | 12.383 | 0.02159 | 1.758 |
| 3 | Bật | 15.092 | 11.993 | 0.02270 | 1.758 |
| 4 | Tắt | 15.198 | 11.910 | 0.00356 | 0 |

Lưu hai ảnh kèm CSV/annotate tốn **12.70–14.43 ms/sự kiện**, chỉ chạy khi có nghi
vấn. Nhánh này không gọi thêm YOLO hoặc OCR. Bộ nhớ ảnh lịch sử tính theo ndarray
được giữ, khử trùng ảnh dùng chung; chưa gồm metadata Python. Peak RSS toàn process
là khoảng 1032.5–1036.1 MiB, có allocator/model cache nên không lấy chênh RSS làm
chi phí RAM thuần của chức năng.

Chi phí kiểm tra trực tiếp nhỏ trong đoạn này. FPS bật thậm chí cao hơn tắt ở
phép đo ngắn, cho thấy biến động hệ thống/warm-up còn đáng kể: **không diễn giải
đó là tăng tốc hoặc cam kết không giảm FPS**. Video dài, độ phân giải và mật độ xe
khác cần đo lại. Dữ liệu thô: [occlusion-validation.json](occlusion-validation.json).

## Kiểm thử tự động

99 test qua, gồm các trường hợp: hồi phục sau 3 frame thiếu detection; timestamp
file/RTSP; hết hạn; UNKNOWN/đổi pha kể cả màu thô chưa qua smoothing; đổi ID/lớp;
dịch chuyển/kích thước bất hợp lý; quay lại; cắt ngoài đoạn; reset phiên; lịch sử
dùng chung ảnh và giải phóng; không OCR khi nghi vấn; hai JPEG, CSV riêng và bbox
cam giữ ở frame kế tiếp. Test cấu hình không còn cố định số camera bằng 2.

RTSP chỉ được kiểm chứng bằng dữ liệu giả, chưa nối camera IP thật. Màu RED được
kiểm tra trên mọi frame đã xử lý, không thể xác minh màu ở frame nguồn bị bỏ.
