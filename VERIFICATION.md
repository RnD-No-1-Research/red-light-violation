# Kết quả kiểm tra bản giao — 2026-09-29

## Video kết quả theo lần chạy

- Giữ cơ chế tự ghi frame đã chú thích; tên MP4 bổ sung camera ID và ngày giờ
  local, vẫn có run_id và số phần để không ghi đè lần trước. Log báo đường dẫn
  lúc mở và lúc đóng, kèm số frame đã ghi.
- **80 tests passed**, Ruff đạt. Test chạy hai lần xác nhận hai MP4 riêng, mỗi
  file đủ 6 frame và đọc được frame cuối. Test yêu cầu dừng sớm xác nhận MP4
  hoàn tất với 3 frame và log có đường dẫn kết quả. Các test dùng video tổng hợp
  và model giả; ghi/đọc MP4 thật bằng OpenCV. Không mô phỏng việc kill tiến trình.

## Weight xe custom `vehicle_n_best.pt` — 30/09/2026

- Đọc metadata trực tiếp: task detect; ID 0=bus, 1=car, 2=motobike, 3=truck.
  Checkpoint có C3k2/C2PSA nên qua bước kiểm tra cấu trúc hiện tại.
- Adapter xe chấp nhận tập con lớp xe thay vì yêu cầu đủ 5 lớp COCO; ánh xạ
  motobike/motorbike thành motorcycle trong kết quả. ID gửi cho YOLO/ByteTrack
  vẫn theo checkpoint; model COCO giữ ID COCO, không bị áp thứ tự custom.
- Giữ cấu hình người dùng: vehicle_model=weights/vehicle_n_best.pt,
  vehicle_conf=0.5, plate_conf=0.8, traffic_light_conf=0.15.
- Toàn bộ test: **78 passed** trên Python 3.12/Ultralytics 8.3.203 CPU. Ruff đạt.
- Pipeline thật chạy đồng thời 32 frame đầu từ mỗi video, giữ kích thước/vạch/ROI
  của nguồn. Hai video output đều có đủ 32 frame và đọc được frame cuối; manager
  trả mã 0. Dữ liệu thử riêng ở output/custom_weight_smoke_1790750673/.
- Đây là kiểm tra tương thích/I/O, không phải benchmark chất lượng hoặc tốc độ
  của model mới. Benchmark YOLO26s dưới đây là lịch sử, không áp dụng cho weight
  custom và các ngưỡng mới. Chưa kiểm thử lại trên Python 3.13 của người dùng.

## Benchmark CPU — 30/09/2026

Đã chạy **9 lượt**: mỗi camera riêng và hai camera đồng thời, mỗi kịch bản 3 lượt. Dùng đúng weights YOLO26s và ngưỡng đèn 0.15/xe 0.25 hiện tại; CPU Ryzen 5 4600H, Python 3.12, PyTorch 2.8.0+cpu, Ultralytics 8.4.165. Tắt cửa sổ xem nhưng giữ tracking, OCR khi có sự kiện và ghi video/bằng chứng/CSV. Tách nạp model và warm-up khỏi FPS.

| Kịch bản | FPS trung vị | Khoảng FPS 3 lượt | Frame mỗi lượt |
|---|---:|---:|---:|
| cam_01 | 5.08 | 4.65–5.10 | 192 |
| cam_02 | 5.90 | 5.77–6.15 | 384 |
| cam_01+cam_02 | 5.58 | 5.43–5.65 | 576 |

**FPS hai camera là tổng, không phải FPS mỗi camera.** Tất cả lượt xử lý đủ frame và kết thúc thành công. Đây là benchmark hiệu năng, chưa có mAP hoặc precision/recall. Đối chiếu ảnh tại giây 50.522 phát hiện một xe bị hai ID tạo hai sự kiện; số dự đoán chưa thể dùng làm số vi phạm đúng.

[Báo cáo chi tiết, latency từng phần và ảnh lỗi ghi trùng](benchmarks/2026-09-30-cpu.md) · [Số liệu thô và cấu hình tái lập](benchmarks/2026-09-30-cpu.json).

Chạy lại trong môi trường có đủ model, video và cache EasyOCR:

```powershell
.\venv\Scripts\python.exe scripts/benchmark.py --repeats 3
```

Lần đo trong công cụ dùng Python 3.12 và nạp mã Ultralytics 8.4.165 từ bản cài local bằng `--ultralytics-source`, vì Python 3.13 không truy cập được. Không coi đây là kết quả trên Python 3.13, GPU hay RTSP; xem báo cáo để tái lập đúng môi trường.


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

## Bổ sung nghi vấn che khuất — 30/09/2026

- Giữ lịch sử gần vạch tối đa 1 giây theo thời gian nguồn; cùng ID quay lại qua
  vạch đúng chiều và đủ điều kiện chỉ tạo nghi vấn, không tạo vi phạm xác nhận.
- Cam_04 ID 82: nhận lại tại 26.621s sau khoảng trống 0.665536s; hai lượt bật đều
  lưu đúng một nghi vấn và ảnh trước/sau, hai lượt tắt không có sự kiện cho ID 82.
- Kiểm tra trực tiếp khoảng 0.022 ms/frame; ảnh lịch sử tối đa 1.758 MiB trong
  181 frame thử nghiệm. FPS pipeline có nhiễu, không cam kết chi phí trên mọi nguồn.
- 99 test qua. Xem [báo cáo và giới hạn](benchmarks/occlusion-validation.md).

## Tái lập

Phục hồi sau bbox biến dạng, cam_03 (01/10/2026): 159 test qua, Ruff đạt.
Ba xe người dùng chỉ ra ở 33,075s (ID trong ảnh 300/417/431) được đối chiếu
thành ID 63/77/84 khi chạy riêng; model thật ghi lần lượt 18,449s / 32,078s /
33,574s. Chạy đủ và giải mã đủ 745 frame cam_03 + cam_04; hai ô tô ID 1/3
cam_04 vẫn không bị ghi nhầm. Phát lại cam_01 giữ nguyên 26 sự kiện và thời điểm.
[Nguyên nhân, ảnh và kết quả kiểm chứng](benchmarks/late-v3-validation.md).

Sửa bỏ sót phát hiện muộn, giữ điểm `y2-20` theo yêu cầu (01/10/2026):
153 test qua, Ruff và kiểm tra YAML đạt. Model thật chạy đủ 1321 frame của 4
camera; MP4 đủ frame và đọc được frame cuối. Cam_01 chạy riêng giải mã đủ 192
frame, ghi 26 sự kiện. Phát lại cùng quan sát với logic cũ ghi 18; 10 giây đầu
tăng 10→14, 5 giây cuối 6→7. Đây là số dự đoán, chưa có precision/recall.
[Báo cáo, ảnh trước/sau và giới hạn](benchmarks/late-v2-validation.md).
Các mốc kiểm thử phía dưới là lịch sử phiên bản trước thay đổi điểm đại diện.

Mặc định phát hiện muộn cho mọi camera (01/10/2026): chuyển sang settings.yaml,
camera mới tự kế thừa, hỗ trợ ghi đè riêng và giữ trạng thái độc lập. 144 test
qua và Ruff đạt; bao gồm hai chiều vạch, vạch dọc/chéo, camera mới/RTSP ở mức cấu
hình và cùng ID ở nhiều camera. [Kiểm chứng đa camera](benchmarks/all-camera-late-validation.md).
Chạy model thật đồng thời bốn video đủ 1321 frame, pipeline trả 0; bốn MP4 đọc
được frame cuối. Sự kiện phát hiện muộn có ở cam_01 và cam_03 trong lượt này.

Cam_03 / phát hiện muộn (01/10/2026): bật quy tắc hiện diện sau vạch theo yêu cầu,
cho phép xe đã dừng, xác nhận 3 frame. Lượt model thật đủ 384 frame ghi ID 61
(xe mũ xanh) tại 17.286s; bbox đỏ, CSV và decision JSON có lý do riêng. ROI đèn
dự phòng cam_03 đã kiểm tra cả pha xanh. 131 test qua.
[Chi tiết và giới hạn](benchmarks/late-detection-validation.md).

Cam_04 / bbox ô tô ID 3 (30/09/2026): thêm bộ lọc biến dạng bbox trước khi ghi
vi phạm/ nghi vấn. 111 test qua; chạy model thật đủ 361 frame không còn ghi ID 3,
giữ ID 86 và nghi vấn ID 82. [Chi tiết và giới hạn](benchmarks/boxmotion-validation.md).

Cam_04 / ID 86 (30/09/2026): thêm ROI dự phòng thân đèn và canvas 256 trong
cameras.yaml. Kiểm tra đủ 361 frame bằng model thật, ghi vi phạm ID 86 tại
33.776s; MP4 đủ frame và giải mã được frame cuối. Nguyên nhân là UNKNOWN ở lúc
cắt vạch, không phải mất track. [Chi tiết](benchmarks/light86-validation.md).

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
