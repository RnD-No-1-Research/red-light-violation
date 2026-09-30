# Phát hiện vi phạm vượt đèn đỏ — YOLO11, MVC, multi-camera

Project Python chạy local trên Windows, nhận nhiều file video hoặc RTSP. Mỗi camera
có vạch dừng, ROI đèn, tracking, video kết quả, ảnh bằng chứng và CSV riêng. Không có web UI.

## Trạng thái weights và phạm vi sử dụng

Không cần có weights sẵn. Chạy `python scripts/download_weights.py` để tải cả ba
checkpoint dưới đây. Script ghim revision và SHA-256, dùng Python chuẩn, không cần
token Hugging Face. Bản ZIP giao kèm đã có ba weights và cache EasyOCR; Git bỏ qua
weights nên mentor clone cần chạy lại script tải.

| Thành phần | Checkpoint và nguồn | Phạm vi |
|---|---|---|
| Xe | [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11), `yolo11n.pt` | COCO: bicycle, car, motorcycle, bus, truck |
| Đèn | [Sawanparuthipqr/trafficlights_detection_models](https://huggingface.co/Sawanparuthipqr/trafficlights_detection_models), `collected_checkpoints/tl-yolo11n.pt` | Bốn nhãn gốc: Green, Off, Red, Yellow |
| Biển | [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection), `license-plate-finetune-v1n.pt` | Phát hiện vùng biển; không tự đọc chữ |
| OCR | EasyOCR, CRAFT + English recognition | Tải tự động lần đầu vào `weights/easyocr/` |

**Khác biệt với yêu cầu checkpoint đúng 3 lớp:** weights đèn công khai có thêm lớp
Off. Project ánh xạ Off → UNKNOWN và xuất RED/GREEN/YELLOW/UNKNOWN; không đổi tên
COCO thành model đèn và không giả lập màu bằng HSV. Code cũng nhận weights tự train
đúng ba lớp qua `traffic_light_labels`. Cách train ba lớp ở cuối README. Chưa có
checkpoint đúng ba lớp đã được kiểm định trên camera Việt Nam trong bản giao này.

Không có đánh giá độ chính xác trên dữ liệu của bạn. Model đèn công khai không có
model card đầy đủ tại thời điểm tải; chưa xác minh dataset và giấy phép riêng của
checkpoint đó. Model biển có cảnh báo rò rỉ train/test trong dataset upstream;
không dùng các chỉ số công bố để hứa hiệu quả ở Việt Nam. Xem thông tin nguồn và
checksum trong `weights/download_manifest.json` và `scripts/download_weights.py`.
Ultralytics và model biển công bố AGPL-3.0; xem giấy phép của từng nguồn trước khi
phân phối hoặc sử dụng ngoài bài thử nghiệm này.

## Yêu cầu và cài đặt trên Windows 11

- Khuyên dùng Python **3.12 64-bit**; code dùng cú pháp Python 3.10+. Python 3.12
  đã được kiểm thử. Các phiên bản Python mới hơn có thể chưa có wheel tương ứng.
- NVIDIA GPU tùy chọn. CPU chạy được nhưng nhiều camera có thể chậm.
- Cần mạng khi cài packages và tải weights lần đầu; cần quyền truy cập mạng camera
  khi dùng RTSP. Không có video giao thông thực tế hoặc URL camera thật đi kèm.

Trong VS Code mở thư mục project, mở terminal PowerShell:

```powershell
# Thay giá trị dưới đây bằng URL repo GitHub thực tế sau khi bạn push.
$RepoUrl = Read-Host 'Nhap URL GitHub cua project'
git clone $RepoUrl red_light_violation
cd red_light_violation
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python scripts/download_weights.py
python scripts/check_models.py
```

Nếu dùng bản ZIP, giải nén và bắt đầu từ lệnh tạo `venv`, bỏ qua clone. Trong VS Code
chọn **Python: Select Interpreter → venv\Scripts\python.exe**. Nếu PowerShell không
cho activate, vẫn chạy được mà không đổi policy hệ thống:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py --check-config
```

`check_models.py` nạp weights thật và gọi thử bốn nhánh AI trên ảnh rỗng. Nó cũng
tải cache EasyOCR nếu chưa có. Lần đầu cần mạng; sau đó đặt
`ocr_download_enabled: false` để yêu cầu cache đã có và chạy offline.

Các đường dẫn trong YAML được giải theo thư mục chứa `main.py`, không phụ thuộc
terminal đang ở đâu. Các tham số CLI `--settings` và `--cameras` tương đối theo cwd.

### Bật CUDA, tùy chọn

Sau khi cài requirements, cài wheel CUDA tương ứng nếu PyTorch hiện tại chỉ có CPU:

```powershell
python -m pip install --force-reinstall --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python scripts/check_models.py
```

Đây là cặp phiên bản trong [hướng dẫn PyTorch 2.8.0](https://pytorch.org/get-started/previous-versions/).
Cần driver NVIDIA tương thích CUDA 12.8. Không bắt buộc cài CUDA Toolkit để chạy
wheel. `device: auto` kiểm tra CUDA và fallback CPU nếu CUDA không dùng được.
`device: cpu` ép CPU. Kiểm thử bản giao chạy CPU; chưa kiểm thử NVIDIA trên máy này.

### Lưu ý OpenCV và EasyOCR

EasyOCR phụ thuộc `opencv-python-headless`, Ultralytics phụ thuộc `opencv-python`.
Requirements ghim cả hai cùng 4.12.0.88; bản cài đã kiểm tra có `WIN32UI` để vẽ vạch.
Nếu môi trường cũ ghi đè binary GUI bằng headless và `imshow` báo không hỗ trợ:

```powershell
python -m pip install --force-reinstall --no-deps opencv-python==4.12.0.88
```

## Chạy kiểm tra nhanh khi chưa có video giao thông

```powershell
python scripts/make_sample_videos.py
python main.py --cameras data/sample/cameras.smoke.yaml
```

Tạo hai clip đồ họa và chạy **model thật** trên hai camera. Kiểm tra đầu ra
`output/smoke_01/` và `output/smoke_02/`. CSV có header; không kỳ vọng vi phạm trên
clip đồ họa. Kiểm tra này chỉ chứng minh nạp model, threading và I/O hoạt động.

## Chuẩn bị hai camera thật

Đặt video ở `data/videos/intersection_01.mp4` và
`data/videos/intersection_02.mp4`, hoặc sửa `source` cho đúng tên. Cấu hình mặc định
bật hai camera file; camera RTSP mẫu tắt. Mỗi mục phải có `camera_id` duy nhất và
chỉ chứa chữ, số, `_`, `-`.

```yaml
cameras:
  - camera_id: cam_01
    name: Video thu nghiem
    source: data/videos/intersection_01.mp4
    stop_line:
      point1: [100, 400]
      point2: [800, 400]
    crossing_direction: negative_to_positive
    traffic_light_roi: [300, 50, 200, 300]
    vehicle_conf: 0.4
    enabled: true
  - camera_id: cam_02
    name: Camera IP
    source: ${CAM_RTSP_URL}
    stop_line:
      point1: [100, 400]
      point2: [800, 400]
    crossing_direction: negative_to_positive
    traffic_light_roi: [300, 50, 200, 300]
    enabled: true
```

Thay URL RTSP trong terminal hiện tại, ví dụ theo cú pháp sau:

```powershell
$env:CAM_RTSP_URL='rtsp://USER:PASSWORD@HOST:554/stream'
```

Thay USER/PASSWORD/HOST bằng thông tin camera thật; URL trên không phải camera demo.
Cũng có thể ghi URL trực tiếp vào YAML local. `.env.example` chỉ là tài liệu mẫu;
project không tự nạp `.env` và không cần `python-dotenv`. Không commit thông tin
đăng nhập thật. Camera tắt không yêu cầu biến môi trường hoặc tọa độ hợp lệ.

Mỗi camera có thể override `traffic_light_conf`, `vehicle_conf`, `plate_conf` ngay
trong mục camera. Các tùy chọn chung khác ở `config/settings.yaml`.

Nếu đèn nhỏ thường bị `UNKNOWN`, có thể cấu hình thêm một ROI dự phòng chỉ chứa
cụm bóng đèn, tránh bảng đếm số. Ví dụ đã hiệu chỉnh cho `vid_2.mp4` 640×480:

```yaml
traffic_light_fallback_roi: [547, 177, 29, 59]
traffic_light_canvas_size: 192
```

Model vẫn xét ROI chính trước. Chỉ khi kết quả là `UNKNOWN`, nó mới thử ROI dự
phòng được đặt giữa nền đen vuông (giữ nguyên pixel, không kéo giãn). Cả hai lượt
dùng cùng model và ngưỡng confidence; kết quả cuối vẫn qua bỏ phiếu 5 frame.
Nếu cả hai lượt không nhận ra đèn, trạng thái vẫn là `UNKNOWN`. Không có cấu hình
dự phòng thì giữ xử lý một lượt như trước. Đệm ảnh không bảo đảm tăng độ chính xác
cho mọi nguồn; cần kiểm tra cả đoạn đỏ, xanh và vàng trước khi dùng.

ROI dự phòng dùng tọa độ frame gốc và hiện chỉnh trong YAML. Khi đổi video, góc
camera hoặc độ phân giải, hãy chỉnh lại hoặc xóa cả hai dòng trên; công cụ vẽ ROI
chính không tự cập nhật ROI dự phòng. Một frame cần dự phòng sẽ tốn thêm một lượt
inference đèn.

### Vẽ vạch, chọn ROI và hướng đi

```powershell
python scripts/draw_stop_line.py --camera_id cam_01
python scripts/draw_stop_line.py --camera_id cam_02
```

1. Script mở frame đầu tiên của đúng nguồn camera, rồi đóng capture.
2. Click hai điểm của đoạn vạch dừng. Click lần thứ ba để chọn lại.
3. Nhấn **R**, kéo vùng chỉ chứa đèn điều khiển làn đang xét, Enter xác nhận ROI.
4. Nhấn **D** nếu cần đổi hướng mũi tên đỏ. Mũi tên chỉ chiều vượt vạch cần bắt.
5. Nhấn **S** lưu vào đúng mục camera; Esc hoặc đóng cửa sổ để hủy.

ROI là `[x, y, width, height]`, vạch và ROI theo pixel **frame gốc**. Script có thể
chỉnh cả camera đang tắt hoặc chưa có vạch hợp lệ; `enabled` được giữ nguyên.
YAML được ghi lại bằng PyYAML nên comment/định dạng có thể thay đổi, giá trị camera
khác được giữ nguyên.

Với vạch trái → phải nằm ngang, `negative_to_positive` bắt xe đi từ trên xuống;
`positive_to_negative` bắt chiều ngược lại. Nếu đường có nhiều hướng, ROI phải ứng
với đúng đèn và đoạn vạch chỉ bao phủ làn đang xét. Không tự suy ra tín hiệu rẽ.

### Chạy hệ thống

```powershell
python main.py --check-config
python main.py
# Hoặc chỉ một camera đang enabled:
python main.py --camera_id cam_01
```

`--check-config` chỉ validate YAML, không xác nhận file tồn tại hay RTSP phản hồi.
Khi chạy thật, mỗi worker mở nguồn, đọc frame, kiểm tra tọa độ nằm trong frame.
Nguồn lỗi/ROI vượt biên chỉ làm dừng camera đó; các camera khác tiếp tục. YAML sai
cấu trúc, trùng ID hoặc thiếu weights làm chương trình báo lỗi cấu hình rõ ràng.

File chạy tới EOF rồi đóng video. RTSP chạy tới Ctrl+C hoặc hết retry. Mỗi lần mất
kết nối: chờ 5 giây và thử tối đa 3 lần; timeout mở/đọc mặc định 5 giây.
Ctrl+C đợi capture timeout và inference hiện tại kết thúc để đóng container MP4.
Mã thoát: 0 thành công, 1 lỗi cấu hình/model hoặc ít nhất một camera lỗi,
130 người dùng ngắt. Khi còn RTSP hoạt động, mã thoát chỉ được trả sau khi dừng.

## Vẽ polygon giới hạn vùng phát hiện xe

Dừng chương trình đang chạy trước khi hiệu chỉnh. Từ terminal của project:

```powershell
.\venv\Scripts\python.exe scripts/draw_polygon.py --camera_id cam_01
.\venv\Scripts\python.exe scripts/draw_polygon.py --camera_id cam_02
```

Click ít nhất **3 đỉnh** theo thứ tự quanh làn/vùng cần xét; thường dùng 4–6 đỉnh.
Polygon cần phủ **cả trước và sau vạch dừng**: xe phải được theo dõi trước lúc cắt
vạch và vẫn ở trong vùng sau khi vi phạm. Vạch đang cấu hình được vẽ màu vàng để
tham chiếu. Không cần đưa cụm đèn vào polygon: ROI đèn được xử lý riêng.

- **S hoặc Enter**: lưu polygon cho đúng camera.
- **R**: xóa các điểm đang vẽ để chọn lại.
- **Backspace hoặc chuột phải**: bỏ đỉnh cuối.
- **Esc hoặc đóng cửa sổ**: hủy, giữ nguyên YAML.

Cửa sổ tự thu nhỏ frame lớn; tọa độ lưu vẫn theo frame gốc. Công cụ giữ nguyên
vạch dừng, ROI đèn, source và cấu hình các camera khác. Chọn polygon đơn, các cạnh
không tự cắt nhau. Cấu hình được lưu dưới khóa `detection_polygon` của từng camera.
Không có khóa này hoặc `detection_polygon: []` nghĩa là nhận xe trên toàn frame.
Để chủ động bỏ vùng đã chọn:

```powershell
.\venv\Scripts\python.exe scripts/draw_polygon.py --camera_id cam_01 --clear
```

Sau khi lưu hai camera, khởi động lại:

```powershell
.\venv\Scripts\python.exe main.py
```

Xe được coi là trong vùng khi **trung điểm cạnh dưới bbox** nằm trong hoặc trên
biên polygon; không yêu cầu toàn bộ bbox lọt trong vùng. Pipeline lọc detection
YOLO trước khi đưa vào ByteTrack và lọc lại vị trí track trước nghiệp vụ/vẽ. YOLO
vẫn suy luận trên frame gốc, nên polygon không bảo đảm tăng FPS. Camera chỉ vẽ và
xét vi phạm đối với xe trong vùng. Viền polygon có màu xanh lam nhạt.

Khi track vi phạm, bbox giữ **màu đỏ và nhãn `VI PHAM`** ở các frame tiếp theo,
kể cả khi đèn đã xanh. Trạng thái này áp dụng cho cùng `track_id`, riêng từng camera,
và không tạo thêm dòng CSV. Ra ngoài polygon thì không vẽ bbox; nếu vào lại với
cùng ID thì vẫn đỏ. Nếu tracker cấp ID mới, reconnect hoặc khởi động lại chương
trình, trạng thái cũ không được tự gán cho ID mới. Không vẽ bbox đứng yên tại vị trí
cũ khi xe đã mất dấu.

## Output và ý nghĩa thời gian

```text
output/cam_01/
├── videos/<run_id>_001.mp4
├── evidence/<event_id>_frame.jpg
├── evidence/<event_id>_vehicle.jpg
├── evidence/<event_id>_plate.jpg
└── violations.csv
```

Tên video/event dùng ID riêng nên không ghi đè lần chạy trước; CSV append. Mỗi
camera có khóa CSV riêng. Không chạy hai tiến trình cùng ghi một thư mục output.
Không có vi phạm thì thư mục evidence rỗng và CSV chỉ có header, đây là bình thường.

CSV có đúng các cột:

```text
id,camera_id,track_id,timestamp,plate_text,plate_confidence,vehicle_class,evidence_path
```

`evidence_path` tương đối với thư mục chứa CSV. Timestamp là UTC ISO-8601 tại lúc
đọc frame trên máy: với video file, đây là giờ xử lý, **không phải giờ ghi hình
thực tế**. Video/bằng chứng có thêm `Video time` là vị trí giây trong file, ước tính
từ chỉ số frame/FPS. Với RTSP, timestamp là lúc nhận frame, không phải đồng hồ IP camera.
`plate_confidence` là confidence OCR trung bình theo số ký tự; bằng 0 khi UNKNOWN.

Full frame có vạch, ROI, polygon nếu có, bbox xe vi phạm tô đỏ, màu đèn và timestamp.
Video và cửa sổ trực tiếp giữ bbox đỏ theo ID sau lần vi phạm.
Crop xe lấy từ ảnh gốc chưa vẽ; crop biển chỉ có khi phát hiện được biển.
OCR lỗi/regex không khớp vẫn ghi full frame/crop xe và CSV UNKNOWN.

## Kiến trúc MVC và threading

```mermaid
flowchart TD
    A[main.py: CLI và YAML] --> B[CameraManager]
    B --> M[Khởi tạo một lần: YOLO đèn, YOLO xe, YOLO biển, EasyOCR]
    B --> C1[Thread camera 01 / ViolationController]
    B --> C2[Thread camera 02 / ViolationController]
    F[File: đọc tuần tự] --> C1
    R[RTSP capture thread: buffer mới nhất] --> C2
    C1 --> M
    C2 --> M
    M --> T1[ByteTrack và lịch sử camera 01]
    M --> T2[ByteTrack và lịch sử camera 02]
    T1 --> V1[View: output/cam_01]
    T2 --> V2[View: output/cam_02]
```

Hướng phụ thuộc: `main → camera_manager → violation_controller → models + utils + views`.
Controller không import torch/ultralytics; View không xét điều kiện vi phạm. Module
`models/runtime.py` chọn device và tạo bundle, rồi manager truyền cùng tham chiếu
cho mọi controller. Các adapter AI có khóa riêng, giữ khóa suốt inference và đọc
kết quả. Inference trên cùng model được tuần tự hóa, nên nhiều thread không đồng
nghĩa đạt realtime với mọi số camera.

`vehicle_detector.py` gọi YOLO detection chung rồi cập nhật **BYTETracker có sẵn
trong Ultralytics** riêng theo camera. Không dùng một `model.track(persist=True)`
chung cho tất cả camera, vì tracker có trạng thái chuỗi frame. Lớp `CameraTracker`
giữ bộ đếm ID tăng dần khi camera khác reset; cập nhật tracker nằm trong khóa chung.
ID không bắt buộc bắt đầu từ 1 cho từng camera. Xem [API BYTETracker](https://docs.ultralytics.com/reference/trackers/byte_tracker/).

Mỗi controller giữ lịch sử bottom-center, cửa sổ màu đèn và tập ID đã ghi riêng.
Chỉ cắt đoạn vạch hữu hạn đúng hướng khi đèn làm mượt là RED mới ghi nhận. Không
ghi xe xuất hiện sẵn phía sau vạch, cắt lúc YELLOW/GREEN/UNKNOWN hoặc lặp cùng ID.
Điểm đúng trên vạch giữ phía quan sát trước đó; chỉ báo khi đã sang phía bên kia.
Track mất một frame xử lý sẽ không được nối vị trí qua khoảng mất dấu để suy ra vi phạm.

Làm mượt cần đủ N frame (mặc định 5) và hơn nửa cửa sổ cùng trạng thái. Hòa phiếu,
chưa đủ cửa sổ hoặc nhiều màu xung đột trong ROI → UNKNOWN. Sau reconnect/thay đổi
độ phân giải, reset tracker và lịch sử để tránh báo dựa trên frame trước gián đoạn.

RTSP producer giữ tối đa một frame mới nhất. Consumer lấy mỗi frame tối đa một lần;
frame cũ bị bỏ khi AI chậm. Video output RTSP ghi các frame đã xử lý theo FPS nguồn,
vì vậy có thể phát nhanh hơn thời gian thực khi rớt frame. Tách file MP4 mới khi
reconnect, thay đổi độ phân giải/FPS. Đây không phải bộ ghi hình RTSP nguyên vẹn.

## Cấu trúc mã nguồn

| Thư mục/tệp | Trách nhiệm |
|---|---|
| `main.py` | CLI, log có camera_id, mã thoát |
| `config/settings.yaml`, `cameras.yaml` | Cấu hình chung và từng camera |
| `models/traffic_light_detector.py` | YOLO đèn trong ROI |
| `models/vehicle_detector.py` | YOLO COCO + ByteTrack riêng theo camera |
| `models/plate_detector.py`, `plate_ocr.py` | Detect biển và EasyOCR |
| `models/_yolo.py`, `runtime.py`, `types.py` | Nạp model có khóa, device, kiểu kết quả |
| `controllers/camera_manager.py` | Khởi tạo model, quản lý worker, Ctrl+C |
| `controllers/violation_controller.py` | Pipeline một camera và nghiệp vụ |
| `views/display.py` | Vẽ, MP4, JPEG, CSV |
| `utils/config_loader.py`, `video_source.py` | Validate cấu hình, file/RTSP/retry |
| `utils/line_crossing.py`, `light_state.py`, `plate_format.py` | Hình học, bỏ phiếu, regex |
| `scripts/draw_stop_line.py` | Hiệu chỉnh vạch/ROI/hướng bằng chuột |
| `scripts/download_weights.py`, `check_models.py` | Tải và kiểm tra AI thật |
| `scripts/make_sample_videos.py`, `data/sample/` | Smoke test hai video tổng hợp |
| `tests/` | Test CPU, không cần weights, video thật hoặc RTSP thật |
| `weights/`, `data/videos/`, `output/` | Tài nguyên local, không commit |

## Lệnh kiểm tra từng phần của Bước 2

Chạy sau khi cài requirements, từ thư mục project:

| Phần | Lệnh |
|---|---|
| 1. Config + nguồn video + CLI | `python main.py --check-config` |
| 2. Bốn adapter AI | `python scripts/check_models.py` |
| 3. Controllers + cắt vạch | `python -m pytest tests/test_line_crossing.py tests/test_multi_camera.py tests/test_tracking.py -q` |
| 4. View, ảnh/CSV/MP4 | `python -m pytest tests/test_multi_camera.py -q` |
| 5. Scripts + tests | `python scripts/draw_stop_line.py --help` rồi `python -m pytest tests/ -q` |
| 6. Dependencies + tổng kiểm | `python -m pip check` rồi `python -m compileall -q main.py models controllers views utils scripts` |

`pytest tests/` cũng chạy được. Test dùng video tổng hợp, fake inference cho nhánh
vi phạm, và ByteTrack thật với boxes tổng hợp; không tải checkpoint hoặc OCR Reader.
Các test I/O dùng thư mục tạm pytest. Không có camera mạng thật trong tests.

## Nếu cần weights đèn đúng ba lớp

Cần tập ảnh giao thông có annotation bounding box ba nhãn red/green/yellow. Dataset
không được đi kèm hoặc tự dựng từ video giả. Tách train/val theo camera/video để
giảm rò rỉ frame gần nhau. Điền đường dẫn tuyệt đối của dataset thật trong
`config/traffic_light_dataset.example.yaml`, rồi từ thư mục project:

```powershell
yolo detect train model=weights/yolo11n.pt data=config/traffic_light_dataset.example.yaml epochs=50 imgsz=640 workers=0 device=cpu project=runs name=traffic_light_3class
```

Nếu CUDA hoạt động, thay `device=cpu` bằng `device=0`. Sau huấn luyện, trỏ
`traffic_light_model` tới `runs/traffic_light_3class/weights/best.pt` và thay mapping:

```yaml
traffic_light_labels:
  red: RED
  green: GREEN
  yellow: YELLOW
```

Số epoch trên là điểm khởi đầu, không bảo đảm chất lượng. Script download không
ghi đè weights có checksum khác, trừ khi bạn chủ động truyền `--force`.

## Hạn chế và xử lý lỗi

- Ban đêm, ngược sáng, đèn nhỏ/nhấp nháy, nhiều đèn trong ROI có thể gây UNKNOWN
  hoặc nhận màu sai. Majority voting có độ trễ, có thể lỡ thời điểm đổi đèn.
- Xe che khuất/đổi ID có thể mất hoặc lặp nhận dạng cùng xe vật lý. Cam kết chống
  trùng theo track ID trong phiên, không nhận dạng cùng xe qua nhiều camera/lần chạy.
- Rung bbox gần vạch có thể tạo cắt vạch giả; cần kiểm tra bằng chứng và hiệu chỉnh
  vạch. Hình học ảnh 2D không chứng minh vị trí bánh xe trên mặt đường.
- Biển Việt Nam hai dòng được sắp theo hàng; biển mờ/bẩn/nghiêng hoặc xe máy xa
  vẫn có thể UNKNOWN. Regex hỗ trợ dạng `29A-12345`, `30E1-12345`, không bao phủ
  biển ngoại giao, quân đội, mọi loại biển cũ; regex không xác nhận biển đã đăng ký.
- RTSP lag, mất frame hoặc nguồn trùng nội dung có thể làm mất thời điểm cắt vạch.
  Retry đã kiểm thử bằng mock; camera IP thật và mạng của bạn chưa được kiểm thử.
- Tập ID đã ghi được giữ đến hết phiên để bảo đảm tối đa một lần; chạy liên tục
  rất lâu với nhiều vi phạm làm tăng bộ nhớ và dung lượng lưu trữ.
- Video không mở: kiểm tra path, codec và quyền đọc. Camera không ghi output:
  xem log camera_id, enabled, ROI/vạch và kết nối. Không có evidence có thể là
  không có vi phạm đủ điều kiện, không phải lỗi ghi file.
- Nếu thiếu weights: `python scripts/download_weights.py`; nếu tải OCR lỗi:
  kiểm tra mạng và chạy lại `python scripts/check_models.py`.
- Chỉ hỗ trợ OpenCV backend FFmpeg cho RTSP, TCP và timeout. Native decoder có thể
  có độ trễ đóng riêng; không có cam kết thời gian shutdown tuyệt đối.

Xem `VERIFICATION.md` để biết những gì thực sự đã kiểm thử trong bản giao.
