# Kết quả kiểm tra bản giao — 2026-09-29

## Bỏ sót xe máy ở cam_02 khoảng giây 50–51

- Kiểm tra video kết quả cũ `5c557bbde497_001.mp4`: xe máy mang ID 974/975 đi
  qua vạch nhưng overlay ghi `LIGHT: UNKNOWN`. Vì điều kiện vi phạm yêu cầu RED
  tại thời điểm cắt vạch, xe không được ghi nhận; nhận RED trở lại sau đó không
  được dùng để kết luận hồi tố.
- Giữ ROI chính/ngưỡng cũ; thêm ROI dự phòng `[547,177,29,59]`, nền đệm 192×192
  riêng cho cam_02. Chỉ chạy lượt dự phòng khi lượt chính trả UNKNOWN. Màu từ
  model vẫn qua cửa sổ bỏ phiếu 5 frame, không ép UNKNOWN thành RED.
- Replay model đèn thật trên đủ 384 frame của `vid_2.mp4`, CPU/Python 3.12,
  Ultralytics 8.3.203. Frame 294–310 (đánh số từ 0, khoảng 48.86–51.52 giây):
  trạng thái cũ UNKNOWN, trạng thái mới RED xuyên suốt. Sau sửa, trạng thái
  frame 109–270 giữ GREEN, không xuất hiện đợt RED giả mới trong đoạn đèn xanh.
- Đây là xác minh riêng nhánh nhận màu đèn, chưa chạy lại toàn bộ YOLO26 + OCR
  trong venv Python 3.13 của người dùng (runtime đó trả Access denied từ môi
  trường công cụ). ID xe có thể khác khi chạy lại.
- Hạn chế còn tồn tại: checkpoint hiện tại có thể nhầm vàng thành đỏ ở đoạn
  chuyển pha khoảng frame 270; lỗi này cũng có ở cấu hình cũ. Cần weights được
  đánh giá/huấn luyện phù hợp để cải thiện phân loại, không coi bản hiệu chỉnh
  này là kiểm chứng độ chính xác toàn hệ thống.
- Test mới kiểm tra bảo toàn pixel khi đệm, UNKNOWN không bị ép thành đỏ,
  ưu tiên màu ROI chính, cấu hình/biên ROI và truyền ROI riêng từng camera.
- Bộ test tích hợp: 74 passed; sau đó bổ sung kiểm tra đọc/biên ROI dự phòng
  và chạy lại toàn bộ test cấu hình: 20 passed. Ruff và `--check-config` đạt.

## Cập nhật polygon và bbox vi phạm

- Bộ test sau cập nhật: **59 passed** trên môi trường kiểm thử Python 3.12.
- Đã kiểm tra polygon lõm, điểm trên biên, tọa độ không hợp lệ, polygon tự cắt,
  cấu hình theo từng camera và tọa độ vượt frame.
- Test YOLO adapter bằng boxes tổng hợp xác nhận loại detection ngoài polygon
  trước khi đưa vào ByteTrack, kể cả khi không còn detection nào.
- Test pipeline xác nhận View không nhận bbox ngoài polygon, không ghi vi phạm
  ngoài vùng, giữ pixel bbox đỏ ở mọi frame sau vi phạm dù đèn chuyển xanh và
  CSV vẫn chỉ có một sự kiện.
- Công cụ `draw_polygon.py` được kiểm tra bằng callback chuột giả: lưu đúng camera
  và giữ nguyên source, vạch, ROI đèn và camera khác. Chưa thử click trực tiếp.
- Phiên pytest có một cảnh báo quyền ghi cache `.pytest_cache` cũ; các test đều đạt.
  Đây là cảnh báo cache kiểm thử, không phải lỗi xử lý camera.

## Đã thực hiện

- Cài dependencies trong một venv riêng trên Windows, Python 3.12.14.
- `pip check`: **No broken requirements found**.
- Tải ba checkpoint bằng URL ghim revision, kiểm tra SHA-256 với manifest upstream.
- Nạp đúng ba YOLO11 và một EasyOCR Reader, gọi inference trên CPU thành công.
  Weights đèn có bốn nhãn Green/Off/Red/Yellow; Off ánh xạ UNKNOWN. Biển có nhãn
  `License_Plate`; model xe có các nhóm xe COCO.
- EasyOCR đã tải CRAFT (`craft_mlt_25k.pth`) và English recognition (`english_g2.pth`).
- Pipeline thật xử lý đồng thời hai file MP4 tổng hợp: mỗi camera xuất MP4 320×240,
  đủ 20 frame và CSV riêng. Đã mở lại video bằng OpenCV và xem frame đầu.
- `python -m pytest tests/ -q`: **45 passed**. Test không cần weights/GPU/RTSP thật.
- Ruff lint và format đạt; `compileall` đạt. Kiểm tra AST không phát hiện function
  thiếu type hint (ngoại trừ self/cls) hoặc public function/class thiếu docstring.
- `main.py --check-config` và `scripts/draw_stop_line.py --help` chạy thành công.
- OpenCV build đang dùng có `WIN32UI`.

## Các tình huống có test tự động

- Vượt đúng hướng lúc đỏ; không báo khi xanh/vàng/UNKNOWN; không báo xe xuất hiện
  sẵn sau vạch; chống ghi trùng; vạch chéo; chạm vạch; ngoài đoạn; mất track.
- Bỏ phiếu cần đủ cửa sổ, hòa phiếu và reset lịch sử.
- Regex biển một/hai dòng; thứ tự các mảnh OCR; không tự sửa ký tự nhầm.
- Hai camera trùng track ID vẫn xuất bằng chứng riêng; lỗi một nguồn không dừng
  camera còn lại; OCR lỗi vẫn lưu UNKNOWN và ảnh; MP4/CSV/JPEG thực sự được tạo.
- ByteTrack thật được cấp boxes tổng hợp: tạo/reset camera B không làm tái sử dụng
  ID đang hoạt động của camera A. Test này không khởi tạo YOLO weights.
- RTSP mock: một lần mở đầu + tối đa ba retry; buffer frame mới nhất không lặp;
  stop ngắt chờ retry và đánh thức consumer.
- Controller mock reconnect: không nối xe trước/sau vạch qua hai phiên; tách MP4.
- Hiệu chỉnh vạch bằng callback chuột giả: lưu đúng camera, giữ nguyên camera khác.
- Validate camera ID, tọa độ, ngưỡng, ROI, trạng thái enabled, biến môi trường,
  YAML sai cú pháp và đường dẫn tương đối theo project.

## Chưa kiểm thử và khác biệt cần biết

- Đã kiểm tra nhánh đèn trên video cam_02 như trên; chưa thử RTSP thật, không có số liệu accuracy,
  precision/recall hoặc cam kết realtime trên camera Việt Nam.
- Chưa kiểm thử inference trên NVIDIA CUDA hoặc kết nối camera IP thật. Reconnect
  được kiểm thử bằng mock; mạng/codec thực tế có thể có hành vi khác.
- Chưa thao tác chuột trực tiếp trong cửa sổ calibration; đã kiểm tra callback lưu
  YAML và backend GUI có mặt. Script cần desktop tương tác để sử dụng thực tế.
- Checkpoint đèn tải về **có 4 lớp, không phải checkpoint đã train đúng 3 lớp**.
  Giao diện nghiệp vụ trả RED/GREEN/YELLOW/UNKNOWN. README có hướng dẫn thay bằng
  weights ba lớp tự train. Không có dataset huấn luyện đi kèm.
- Video smoke chỉ là đồ họa kiểm tra I/O. Nhánh tạo vi phạm được kiểm thử bằng fake
  AI có kiểm soát, không được trình bày như kết quả nhận diện giao thông thật.

## Tái lập

Từ thư mục project đã cài requirements:

```powershell
python scripts/download_weights.py
python scripts/check_models.py
python -m pytest tests/ -q
python scripts/make_sample_videos.py
python main.py --cameras data/sample/cameras.smoke.yaml
```

Để sử dụng thực tế: đặt video hoặc RTSP thật, bật camera, dùng
`python scripts/draw_stop_line.py --camera_id cam_01` cho từng camera rồi chạy
`python main.py`. `output/`, video đầu vào và weights được loại khỏi Git.
