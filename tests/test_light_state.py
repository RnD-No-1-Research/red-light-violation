"""Kiểm tra bỏ phiếu và chống sử dụng đèn cũ sau reconnect."""

from utils.light_state import LightSmoother


def test_majority_warmup_and_reset() -> None:
    """Cần đủ cửa sổ; UNKNOWN tham gia bỏ phiếu như các trạng thái khác."""
    smoother = LightSmoother(5)
    assert [smoother.update(s) for s in ["RED", "RED", "GREEN", "UNKNOWN"]] == [
        "UNKNOWN"
    ] * 4
    assert smoother.update("RED") == "RED"
    smoother.reset()
    assert smoother.update("RED") == "UNKNOWN"


def test_tie_is_unknown() -> None:
    """Không chọn RED tùy ý khi số phiếu bằng nhau."""
    smoother = LightSmoother(4)
    for color in ["RED", "GREEN", "RED"]:
        smoother.update(color)
    assert smoother.update("GREEN") == "UNKNOWN"
