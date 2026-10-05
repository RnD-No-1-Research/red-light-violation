# Phát hiện vi phạm vượt đèn đỏ — YOLO26, multi-camera

#Bài toán và hướng giải quyết

Bài toán phát hiện phương tiện vi phạm vượt đèn đỏ từ đầu vào là file mp4 hoặc RTSP ( chưa kiểm thử) 
Có vùng polygon theo dõi xe, chỉ xe trong polygon mới được detect và xác định vi phạm, vạch xác định vi phạm và chiều đi của làn , ROI để xác định đèn giao thông và màu đèn

Hướng giải quyết: 
1. Nhận diện dữ liệu và khởi tạo camera

   Chương trình đọc settings.yaml và cameras.yaml để lấy nguồn video hoặc RTSP, weights, ngưỡng confidence, ROI đèn, polygon và vạch dừng của từng cam
   
2. Nhận diện màu đèn trong vùng ROI

   Từ frame gốc, chương trình cắt ROI đèn và chạy model nhận diện màu đèn: RED, YELLOW, GREEN, UNKNOWN.
   
3. Phát hiện phương tiện và lọc vùng xét
   
   Model vehicle_n_best.pt dùng để nhận diện phương tiện chạy trên từng frame, trả về bounding box, confidence và lớp phương tiện.
   
   Chương trình giữ các detection có điểm dưới box nằm trong polygon, rồi chuyển sang ByteTrack
   
4. Tracking phương tiện
   
   ByteTrack liên kết detection hiện tại với các track đang có để duy trì ID
   
   Mỗi xe sẽ có đầu ra gồm : track_id, vehicle_class, confidence, bbox
5. Xác định vi phạm
   
   Với một Bbox thì sẽ có điểm đại diện là P = ((x1 + x2) / 2, y2)
   
   Với vạch dừng đã xác định trước qua 2 điểm A,B, tính : s(P) = (Bx − Ax)(Py − Ay) − (By − Ay)(Px − Ax) để xác định chiều đi của xe, s(P)>0 thì là xe ở trên vạch, s(P)<0 thì là xe ở dưới vạch, s(P)=0 thì là xe nằm trên vạch, yếu tố này để xác định chiều đi của xe, tránh trường hợp có phương tiện ngược chiều đi qua vạch và bị nhận là vi phạm vượt đèn đỏ
   
6. Phân loại kết quả
   
   TH1: Vi phạm trực tiếp, xe có cùng ID ở các frame liên tiếp, cắt vạch theo chiều quy định, đèn được detect là RED. Kết quả là tạo bbox đở và event vi phạm

 <img width="1267" height="812" alt="image" src="https://github.com/user-attachments/assets/9ed00eec-16c3-4050-9301-77c0ce813790" />

   TH2: Nghi vấn do mất track, xe có cùng ID, xuất hiện lại trong tối đa 1s , đèn được detect là RED. Kết quả tạo bbox cam, lưu evidence để xem lại. ( trường hợp này xảy ra khi 2 xe đi quá sát nhau dẫn tới trường hợp ngộp bbox và làm mất ID của 1 xe)

   TH3: Phương tiện xuất hiện sẵn sau vạch khi bắt đầu camera và đèn được báo là RED: Lúc này sẽ xét trường hợp xem đèn đã ổn định là RED được từ một giấy trở lên hay chưa, và xét hướng di chuyển của xe cùng hướng với chiều xét vi phạm hay không, nếu cả 2 điều kiện trên được thoả mãn, thì xe sẽ bị xét là vi phạm, hiện box đỏ và bắn event

   TH4: Xe được phát hiện muộn, đứng sau vạch nhưng chưa biết hướng : Lúc này sẽ hiện box màu cam sau đó từ hướng chuyển động của xe để xác định trường hợp chính xác 

   <img width="1262" height="807" alt="image" src="https://github.com/user-attachments/assets/7a33b93e-5b2d-4c12-819d-7bb8d3d43a3d" />

   
   TH5: Không đủ điều kiện, xảy ra khi phương tiện đi qua vạch sai chiều quy định hoặc ngoài vùng polygon detect, kết quả detect đèn hiện UNKNOWN. Kết quả là không tạo event
   
   Với các xe vi phạm thì sẽ crop xe và lưu lại        

#Các trường hợp khác có thể xảy ra 
1. Xe đi ngược chiều khi đèn đỏ và đi qua vạch : Vẫn detect xe nhưng không xác định vi phạm do không đi đúng chiều quy định
2. Mất track sau khi phương tiện vi phạm đi qua vạch : Không xác nhận vi phạm
3. Xe xuất hiện sẵn sau vạch : Không xác nhận vi phạm do xe không có chuyển động đi qua vạch
4. Trường hợp xe đi ngang qua vùng polygon nhận diện hoặc xe đi từ hướng khác vào vùng polygon nhận diện khi đèn vẫn đang đỏ: Vẫn detect bình thường những không hiện box đỏ và event do không tính là vượt đèn đỏ
#Chức năng của các file chính
main.py: Đọc tham số chạy, cấu hình và khởi động

controllers/camera_manager.py:	Quản lý worker từng camera và hiển thị

controllers/violation_controller.py:	Điều phối xử lý mỗi frame, quyết định nhánh sự kiện

utils/video_source.py: Đọc video/RTSP, thời gian và phiên nguồn

models/vehicle_detector.py: YOLO phương tiện, lọc polygon, ByteTrack

models/traffic_light_detector.py + utils/light_state.py: Nhận diện và làm mượt màu đèn

utils/line_crossing.py + utils/box_motion.py: Kiểm tra cắt vạch và biến dạng bbox

utils/occlusion_review.py:	Xét nghi vấn khi mất dấu ngắn

views/display.py:	Vẽ bbox, ghi video, ảnh và CSV


#BenchMark 
1. CPU:AMD Ryzen 5 4600H with Radeon Graphics
2. GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
3. Python: 3.13.15
4. Thư viện: PyTorch 2.6.0+cu124; Ultralytics 8.4.165; OpenCV 4.12.0; EasyOCR 1.7.2
5. Model: yolo26s.pt, traffic_light.pt
6. Confidence: 0.15 đối với đèn giao thông, 0.5 đối với phương tiện
7. Tốc độ inference: 25.52 ms/frame
8. Fps: 39.18
9. Số Frame đo: 90
   
