# Phục hồi sau bbox lỗi — cam_03, 01/10/2026

## Ba xe người dùng chỉ ra

Ảnh người dùng ở thời gian video **33,075s**, cam_03, có ba bbox xanh sau vạch:
ID 300, 417, 431. Tái lập bằng cùng video/weights/cấu hình và so vị trí bbox tại
frame 199. ID thay đổi khi chạy camera riêng, nên bảng dưới ghi cả hai bộ ID.

| ID trong ảnh | ID chạy riêng | Nguyên nhân cũ | Thời gian ghi ở bản sửa |
|---|---:|---|---:|
| 300 | 63 | Cắt vạch tại 18,117s bị chặn `box_deformation`; nhánh muộn xóa quyền xét lại | 18,449s |
| 417 | 77 | Đáy bbox nhảy lùi tại 31,413s, ứng viên bị loại; bbox ổn định lại vẫn không được xét | 32,078s |
| 431 | 84 | Cắt vạch tại 33,075s bị chặn `box_deformation`; các frame sau bị loại theo ID | 33,574s |

Với xe 431, **không đổi bbox thành đỏ ngay tại 33,075s**: lúc đó chưa có đủ
chuyển động đáng tin sau lần co bbox. Bản sửa ghi sau khoảng 0,499s, khi quan sát
tiếp theo đủ điều kiện. Không sửa lại video quá khứ hoặc giả định đã thấy một
lần cắt vạch đáng tin.

- [Trước sửa tại 33,075s](late-v3-before-33075.jpg).
- [Sau sửa tại 33,075s: hai xe đã đỏ](late-v3-after-33075.jpg).
- [Sau sửa tại 33,574s: đủ cả ba xe đỏ](late-v3-after-33574.jpg).

## Thay đổi code và nguyên tắc bảo vệ

`controllers/violation_controller.py` truyền lý do từ chối của nhánh trực tiếp
cho `utils/late_detection.py`, thay vì chỉ truyền tập ID bị chặn.

- Với `box_deformation`, ID có lịch sử hợp lệ được chuyển sang
  `recovering_motion`. Cắt vạch lỗi vẫn bị từ chối.
- Với đáy bbox nhảy lùi hoặc biến dạng trong chuỗi ứng viên, xóa tiến độ cũ,
  lấy bbox hiện tại làm mốc mới trong dải sau vạch.
- Chuỗi phục hồi bắt buộc có chuyển động tiến ở ít nhất 2 quan sát, đủ ngưỡng
  8% chiều cao bbox (tối thiểu 2 pixel), đủ tỷ lệ hướng đi và vượt qua kiểm tra
  chuyển động mép trên/mép dưới. Không cộng độ dịch do lần co/gộp bbox ban đầu.
- **Đứng yên không đủ để phục hồi**, dù cấu hình chung cho phép xe mới xuất hiện
  đứng yên sau vạch. Cơ chế này tránh ghi bù ô tô chỉ bị co đáy bbox.
- Đổi lớp, kích thước nhảy quá lớn, số frame không tăng hoặc ra ngoài đoạn
  vạch vẫn bị loại. Xanh/vàng xóa trạng thái; reconnect reset từng camera.
- Sự kiện phục hồi vẫn là `LATE_DETECTION_AFTER_LINE`,
  `observed_crossing: false`, thêm `recovered_after_box_rejection: true`.

Không hard-code ID hoặc camera. Giữ điểm `y2-20`, vạch/ROI/polygon và cấu hình
người dùng. Không thêm model hoặc lượt YOLO phương tiện.

## Kiểm chứng

Runtime: Python 3.12, torch 2.8 CPU, mã Ultralytics 8.4.165 từ bản cài local.
Model phương tiện `vehicle_n_best.pt`; ngưỡng xe 0,3, đèn 0,25; imgsz 640.

1. Tái lập lỗi trên toàn bộ 384 frame cam_03; lưu tracks, màu đèn, quyết định và
   ID bị chặn. Phát lại cùng quan sát với logic trước/sau: số dự đoán 10→14.
2. Chạy model thật sau sửa cho cam_03 và cam_04: 384 + 361 = **745 frame**.
   Đã giải mã đủ mọi frame của hai MP4, kiểm tra bằng chứng và metadata, không
   có ID trùng trong CSV của mỗi lượt.
3. Cam_03 ghi 14 sự kiện. Toàn bộ ID và frame sự kiện khớp kết quả phát lại;
   ba xe trong bảng đều có metadata phục hồi.
4. Cam_04 ghi 7 sự kiện. Cả ô tô ID 1 và ID 3 không nằm trong danh sách vi phạm;
   [ảnh tại 36,771s](late-v3-car-check.jpg) vẫn xanh. Đây là kiểm chứng cụ thể,
   không khẳng định loại hết mọi trường hợp báo nhầm do bbox.
5. Phát lại đủ 192 frame quan sát cam_01: giữ nguyên 26 ID, thời điểm và loại
   sự kiện của bản sửa trước.
6. **159 pytest đạt**, Ruff đạt. Các test mới dùng quỹ đạo ba xe thực, ô tô co
   bbox rồi đứng yên/jitter qua nhiều lần hết hạn, đổi lớp và nhảy kích thước.

[Kết quả JSON và đường dẫn video](late-v3-validation.json).

Video mới:

```text
output/late_v3_20261001_134939/cam_03/videos/cam_03_20261001_134941_1a6fa3a720ec_001.mp4
output/late_v3_20261001_134939/cam_04/videos/cam_04_20261001_135019_7f489f30f713_001.mp4
```

Đây là kiểm chứng hồi quy theo các trường hợp người dùng cung cấp, chưa có
precision/recall đã gán nhãn. Phục hồi là suy luận từ chuyển động mới sau vạch;
ID switch hoặc bbox sai nhưng chuyển động hợp lý vẫn có thể gây báo nhầm.
Xe không còn đủ quan sát mới có thể tiếp tục bị bỏ sót. Không có benchmark FPS
so sánh trong lượt này; thêm sự kiện có thể làm tăng số lần OCR và ghi ảnh.
