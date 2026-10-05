# Quy tắc phát hiện muộn dùng chung cho mọi camera — 01/10/2026

`settings.yaml` chứa mặc định: bật, dải sau vạch một chiều cao bbox, đèn đỏ ổn
định ít nhất một giây, chấp nhận xe đã dừng sau vạch. Không cần cờ bật riêng cho
cam_03. Camera mới (file hoặc RTSP) tự kế thừa; bốn trường tương ứng trong mỗi
camera chỉ có tác dụng ghi đè khi được khai báo.

Thứ tự ưu tiên: cấu hình camera → settings chung → mặc định code. False là một
giá trị ghi đè hợp lệ, không bị coi là thiếu cấu hình. Giá trị số/bool không hợp
lệ được báo lỗi khi đọc YAML. Controller nhận cấu hình đã kết hợp rồi tạo bộ
trạng thái riêng; không có nhánh đặc biệt theo tên camera/track ID.

Vạch, chiều đi, polygon và ROI đèn luôn lấy theo camera hiện tại. Khoảng cách
sau vạch tính theo pháp tuyến của vạch và chiều cao bbox, không lấy tọa độ ví dụ
cam_03 làm tọa độ chung. ROI đèn dự phòng vẫn cần hiệu chỉnh theo góc nhìn.

## Kiểm tra tự động

144 test qua, Ruff đạt, CLI kiểm tra cấu hình nhận đủ cam_01..cam_04. Các test
bổ sung kiểm tra camera hiện có, camera mới, cấu hình nguồn RTSP, mặc định khi
thiếu khóa, ghi đè bật/tắt và ngưỡng, cả hai chiều cắt, vạch chéo/dọc, cùng ID
ở nhiều camera và reset một camera không xóa trạng thái camera khác.

## Kiểm chứng với model thật

Chạy đồng thời bốn nguồn video bằng CameraManager, tắt GUI nhưng giữ toàn bộ
pipeline, OCR khi có sự kiện và ghi MP4/JPEG/CSV/decision JSON. Dùng ngưỡng xe
**0.3 đang có trong cấu hình người dùng**, không dùng lại ngưỡng 0.5 của các lượt
kiểm chứng trước. Runtime Python 3.12, torch CPU 2 threads, source Ultralytics
8.4.165; chưa kiểm chứng runtime Python 3.13/CUDA hoặc kết nối RTSP thật.

Kết quả và đường dẫn video: [all-camera-late-validation.json](all-camera-late-validation.json).

Pipeline trả mã 0, xử lý đủ **1321 frame**. Cả bốn MP4 đọc được frame cuối, các
đường dẫn bằng chứng/decision JSON hợp lệ, không trùng track ID trong CSV mỗi
phiên camera. Thời gian lượt chạy khoảng 176.5s (kiểm chứng chức năng, không phải
benchmark so sánh trước/sau).

| Camera | Frame đầu vào/đầu ra | Sự kiện dự đoán | Trong đó phát hiện muộn |
|---|---:|---:|---:|
| cam_01 | 192/192 | 15 | 1 |
| cam_02 | 384/384 | 14 | 0 |
| cam_03 | 384/384 | 9 | 3 |
| cam_04 | 361/361 | 7 | 0 |

Không phát sinh sự kiện muộn trong một clip không có nghĩa nhánh bị tắt: cấu
hình hiệu lực trong JSON xác nhận cả bốn camera cùng bật và chấp nhận hiện diện
sau vạch. Sự kiện chỉ tạo khi dữ liệu thỏa điều kiện.

PowerShell wrapper báo NativeCommandError khi chuyển log stderr INFO của Python
vào file; pipeline hoàn thành mã 0 và các assertion đọc MP4/CSV/JSON đều qua.

Số sự kiện trong JSON là đầu ra dự đoán, không phải số vi phạm đúng đã gán nhãn.
Kiểm chứng hoạt động ở nhiều camera không chứng minh cùng độ chính xác trên mọi
góc nhìn, mật độ xe hoặc nguồn mới. Quy tắc hiện diện sau vạch vẫn có thể nhầm xe
đã vào từ pha xanh, xe từ hướng khác hoặc ID mới của xe cũ; metadata lưu riêng lý
do suy luận để có thể đối chiếu.
