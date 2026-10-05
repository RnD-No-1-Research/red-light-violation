# Kiểm tra mất bbox khi hai xe máy chồng nhau — cam_04

Nguồn: `data/videos/vid_4.mp4`, 640×480, khoảng 6.010 frame/s. Kiểm tra lại
211 frame đầu với `vehicle_n_best.pt`, imgsz=640, ngưỡng xe=0.5, đèn=0.15,
Ultralytics 8.4.165 trên Python 3.12/CPU. Giữ nguyên polygon, vạch và tracking.
Đây là phân tích một tình huống, không phải benchmark độ chính xác tổng thể.

## Quan sát (frame đánh số từ 0)

| Frame | Giây | Đèn đã làm mượt | Track 82 | Track 83 |
|---:|---:|---|---|---|
| 156 | 25.96 | RED | Bbox riêng, điểm dưới (276,396), trước vạch | Có bbox riêng |
| 157 | 26.12 | RED | Mất bbox | Một bbox bao vùng hai xe |
| 158 | 26.29 | RED | Mất bbox | Một bbox bao vùng hai xe |
| 159 | 26.46 | RED | Mất bbox | Một bbox bao vùng hai xe |
| 160 | 26.62 | RED | Tìm lại cùng ID, điểm dưới (309,370), vừa sang phía kia | Có bbox riêng |
| 161 | 26.79 | RED | Điểm dưới (315.5,367), tiếp tục sau vạch | Có bbox riêng |

Vạch hiện tại A=(193,369), B=(638,374), chiều dương→âm. Giá trị signed_side
của ID 82 tại frame 156 là +11600; tại frame 160 là -135. Điểm ở frame 160
chỉ mới qua đường vạch một khoảng nhỏ; các frame tiếp theo xác nhận xe tiếp tục
đi sang phía kia. Hai lần quan sát cách nhau 4 bước, khoảng 0.67 giây; có 3 frame
không có quan sát riêng của xe 82.

Đầu ra YOLO trước ByteTrack ở frame 157–159 chỉ có một bbox tại vùng hai xe,
confidence lần lượt khoảng 0.818, 0.842, 0.804. Không phải chỉ do ngưỡng tạo track
0.5 loại một detection yếu. ByteTrack đã giữ và tìm lại ID 82 trong buffer hiện có.

## Thử nghiệm riêng nhánh detection

Trên frame 156–160, hạ confidence xuống 0.05 để kiểm tra detection yếu:

| Biến thể | Frame 157–159 |
|---|---|
| imgsz=640, NMS IoU=0.7 | Vẫn chỉ một bbox tại vùng hai xe |
| imgsz=640, NMS IoU=0.85 | Vẫn chỉ một bbox; frame 156 xuất hiện thêm bbox trùng |
| imgsz=960, NMS IoU=0.7 | Vẫn chỉ một bbox tại vùng hai xe |

Các thử nghiệm không đổi cấu hình vận hành của project. Không kết luận những
tham số này không hữu ích ở mọi video; chỉ chưa giải quyết được tình huống này.

## Nguyên nhân không tạo sự kiện

`LineCrossing.update()` yêu cầu quan sát liên tiếp; `prune()` cũng loại lịch sử
cũ khi mất dấu. Vì vậy tại frame 160, dù cùng ID trở lại và đèn RED, hệ thống
không dùng vị trí frame 156 để xét cắt vạch. Tăng track_buffer không thay đổi
điều kiện nghiệp vụ này. Giới hạn gap 0.5 giây cũng không đủ cho khoảng 0.67 giây.

Hướng cải thiện cần kiểm thử: giữ lịch sử qua khoảng mất dấu ngắn, kiểm tra
thời gian nguồn, cùng ID, chuyển động phù hợp, giao đoạn đúng chiều và lịch sử
đèn RED suốt khoảng mất dấu. Với trường hợp này cần thử cửa sổ ít nhất khoảng
0.7 giây, chẳng hạn 0.8–1.0 giây, rồi đo báo nhầm/đổi ID trên cả các tình huống
khác. Không kết luận vi phạm chỉ từ bbox dự đoán lúc xe đang bị che; nếu danh tính
hoặc thời điểm chuyển đèn không rõ thì cần đánh dấu sự kiện nghi vấn để kiểm tra.
Chưa sửa logic vượt vạch hoặc tham số tracking trong lần chẩn đoán này.
