# Video mẫu kiểm tra I/O

Chạy từ thư mục project:

```powershell
python scripts/make_sample_videos.py
python main.py --cameras data/sample/cameras.smoke.yaml
```

Hai clip 2 giây chỉ chứa chữ và hình chữ nhật được sinh bằng OpenCV.
Pipeline dùng weights thật, tạo video output và CSV riêng cho hai camera.
Không kỳ vọng phát hiện xe, biển hoặc vi phạm trên đồ họa này. Đây không phải
video giao thông và không phải bằng chứng về độ chính xác AI.

Để kiểm tra toàn bộ nhánh lưu vi phạm một cách xác định, chạy
`python -m pytest tests/test_multi_camera.py -q`. Test đó dùng model giả và
video/ảnh/CSV thật trong thư mục tạm, không thay đổi pipeline chạy chính.

