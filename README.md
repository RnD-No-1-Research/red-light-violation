# Phát hiện vi phạm vượt đèn đỏ — YOLO26, MVC, multi-camera

#Bài toán và hướng giải quyết

Bài toán phát hiện phương tiện vi phạm vượt đèn đỏ từ đầu vào là file mp4 hoặc RTSP ( chưa kiểm thử) 
Có vùng polygon theo dõi xe, chỉ xe trong polygon mới được detect và xác định vi phạm, vạch xác định vi phạm và chiều đi của làn , ROI để xác định đèn giao thông và màu đèn

Hướng giải quyết: 
1. Detect xe, lọc theo polygon và dùng ByteTrack để track các xe
2. Nhận diện màu đèn trong vùng ROI
3. Dùng trung điểm cạnh dưới của BBox xe làm điểm đại diện, kiểm tra nếu điểm này đi qua vạch và đúng chiều của làn để xác định vi phạm
4. Nếu phương tiện vượt qua vạch khi đèn đang được xác định là RED thì sẽ tạo event, rồi BBOX xe sẽ chuyển sang màu đỏ

#Các trường hợp xảy ra 
1. Xe đi ngược chiều khi đèn đỏ và đi qua vạch : Vẫn detect xe nhưng không xác định vi phạm do không đi đúng chiều quy định
2. Mất track sau khi phương tiện vi phạm đi qua vạch : Không xác nhận vi phạm
3. Xe xuất hiện sẵn sau vạch : Không xác nhận vi phạm do xe không có chuyển động đi qua vạch
4. 

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
   
