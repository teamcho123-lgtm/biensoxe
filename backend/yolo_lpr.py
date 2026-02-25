#!/usr/bin/env python3
# Khai báo trình thông dịch Python (hữu ích khi chạy trên Linux/macOS)

import os
# Thư viện thao tác với hệ thống file, thư mục

import cv2
# OpenCV: xử lý ảnh, biến đổi hình học, threshold, contour...

import numpy as np
# NumPy: xử lý mảng số, tính toán hình học

import easyocr
# EasyOCR: nhận diện ký tự (OCR) từ ảnh biển số

import re
# Regex: kiểm tra và chuẩn hóa định dạng biển số

from ultralytics import YOLO
# Thư viện YOLOv8 (Ultralytics) dùng để phát hiện biển số

# ----------------- Cấu hình -----------------
YOLO_WEIGHTS = "best.pt"
# File trọng số YOLO đã huấn luyện để detect biển số xe

# EasyOCR reader: 'en' được khuyến nghị để tách chữ và số
READER_GPU = False  # đặt True nếu bạn có GPU + môi trường CUDA đầy đủ
# Bật/tắt GPU cho EasyOCR (False để chạy CPU cho ổn định)

ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
# Chỉ cho phép OCR nhận các ký tự hợp lệ trong biển số

MAG_RATIOS = [1.0, 1.6]   # tỷ lệ phóng ảnh thử cho ảnh crop nhỏ
# OCR chạy nhiều scale để tăng khả năng đọc ảnh nhỏ/mờ

PAD_RATIO = 0.04
# Tỷ lệ padding khi crop biển số để tránh mất viền

MIN_AREA = 120
# Diện tích tối thiểu để loại bỏ vùng nhiễu

# Bảng chuyển ký tự dễ nhầm lẫn
CONFUSION_MAP = {
    "B": "8", "8": "8",
    "O": "0", "Q": "0", "D": "0", "0": "0",
    "I": "1", "L": "1", "1": "1",
    "Z": "2", "S": "5", "5": "5",
}
# Ánh xạ các ký tự dễ bị OCR nhầm sang dạng chuẩn

# Regex kiểm tra định dạng biển số VN (phiên bản đơn giản)
PLATE_REGEXES = [
    re.compile(r"^\d{2}[A-Z]-\d{3}\.\d{2}$"),  # Ví dụ: 30F-441.01
    re.compile(r"^\d{2}-\d{5}$"),              # Ví dụ: 12-34567
    re.compile(r"^\d{2}[A-Z]-\d{4}$"),
    re.compile(r"^\d{2}[A-Z]\d{5}$"),
]
# Danh sách các mẫu định dạng biển số Việt Nam phổ biến

PROVINCE_CODES = {f"{i:02d}" for i in range(1,100)}
# Tập hợp mã tỉnh hợp lệ từ 01 đến 99

# ----------------- Khởi tạo mô hình -----------------
try:
    detector = YOLO(YOLO_WEIGHTS)
    # Load mô hình YOLO với file trọng số đã huấn luyện
except Exception as e:
    print(f"Lỗi khi tải mô hình YOLO '{YOLO_WEIGHTS}': {e}")
    # In thông báo lỗi nếu không load được model
    print("Hãy chắc chắn best.pt nằm cùng thư mục với file Python.")
    # Gợi ý cách khắc phục
    raise SystemExit(1)
    # Thoát chương trình nếu lỗi nghiêm trọng

reader = easyocr.Reader(['en'], gpu=READER_GPU)
# Khởi tạo EasyOCR (dùng ngôn ngữ English để nhận chữ + số)

# ----------------- Hàm xử lý ảnh -----------------
def safe_pad_and_crop(img, xyxy, pad_ratio=PAD_RATIO):
    """Crop biển số kèm pad an toàn để tránh mất góc."""
    # Hàm cắt vùng biển số từ ảnh gốc và thêm padding
    h, w = img.shape[:2]
    # Lấy chiều cao và chiều rộng ảnh
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    # Tọa độ bounding box YOLO
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)
    # Chiều rộng và chiều cao bounding box
    pad_x = int(bw * pad_ratio); pad_y = int(bh * pad_ratio)
    # Tính padding theo tỷ lệ
    xa = max(0, x1 - pad_x); ya = max(0, y1 - pad_y)
    # Giới hạn tọa độ trái/trên
    xb = min(w - 1, x2 + pad_x); yb = min(h - 1, y2 + pad_y)
    # Giới hạn tọa độ phải/dưới
    return img[ya:yb, xa:xb].copy(), (xa, ya, xb, yb)
    # Trả về ảnh crop và tọa độ mới

def rectify_plate(crop):
    """Chỉnh nghiêng biển số bằng contour lớn nhất + phép biến đổi phối cảnh."""
    # Hàm làm thẳng biển số bị nghiêng
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Chuyển ảnh sang grayscale
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        # Làm mờ để giảm nhiễu
        thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 25, 8)
        # Nhị phân hóa thích nghi
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # Kernel cho phép toán hình thái
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
        # Đóng các lỗ nhỏ trong ảnh
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Tìm contour ngoài
        if not contours:
            return crop
        # Nếu không có contour thì trả ảnh gốc
        c = max(contours, key=cv2.contourArea)
        # Lấy contour lớn nhất
        rect = cv2.minAreaRect(c)
        # Tìm hình chữ nhật bao nhỏ nhất
        box = cv2.boxPoints(rect).astype("float32")
        # Lấy 4 góc hình chữ nhật

        # Tìm 4 góc TL TR BR BL
        s = box.sum(axis=1); diff = np.diff(box, axis=1).reshape(-1)
        # Tổng và hiệu tọa độ để phân biệt vị trí góc
        tl = box[np.argmin(s)]; br = box[np.argmax(s)]
        tr = box[np.argmin(diff)]; bl = box[np.argmax(diff)]
        # Xác định thứ tự các góc
        src = np.array([tl, tr, br, bl], dtype="float32")
        # Tọa độ nguồn

        wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
        hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
        # Tính chiều rộng và chiều cao
        W = max(int(wA), int(wB)); H = max(int(hA), int(hB))
        # Kích thước ảnh sau khi chỉnh
        if W < 10 or H < 8:
            return crop
        # Nếu biển quá nhỏ thì bỏ qua

        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype="float32")
        # Tọa độ đích sau khi biến đổi
        M = cv2.getPerspectiveTransform(src, dst)
        # Ma trận biến đổi phối cảnh
        warp = cv2.warpPerspective(crop, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        # Thực hiện biến đổi ảnh

        # Biển số đứng thì xoay lại
        if warp.shape[1] < warp.shape[0]:
            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
        return warp
    except Exception:
        return crop
        # Nếu lỗi thì trả về ảnh gốc

def remove_plate_border(crop):
    """Cắt bỏ viền biển số để OCR sạch hơn."""
    # Hàm loại bỏ khung viền biển số
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Chuyển sang grayscale
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Nhị phân hóa bằng OTSU
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Tìm contour
        if not contours:
            return crop
        # Nếu không tìm được contour
        c = max(contours, key=cv2.contourArea)
        # Lấy contour lớn nhất
        x,y,w,h = cv2.boundingRect(c)
        # Bounding box của contour
        H, W = crop.shape[:2]
        # Kích thước ảnh crop
        if w < 0.6 * W or h < 0.6 * H:
            # Nếu contour chưa chiếm phần lớn ảnh
            pad = 2
            # Padding nhỏ
            xa = max(0, x-pad); ya = max(0, y-pad)
            xb = min(W, x+w+pad); yb = min(H, y+h+pad)
            return crop[ya:yb, xa:xb].copy()
            # Cắt lại vùng chứa biển số
        return crop
    except Exception:
        return crop
        # Nếu lỗi thì trả về ảnh ban đầu
def preprocess_variants(crop):
    """Sinh các phiên bản tiền xử lý ảnh để chạy OCR tốt hơn."""
    # Hàm tạo ra nhiều phiên bản xử lý ảnh khác nhau từ ảnh biển số

    variants = []
    # Danh sách lưu các ảnh đã xử lý kèm mô tả

    variants.append((crop, "orig"))
    # Thêm ảnh gốc, không xử lý

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Chuyển ảnh sang thang xám để xử lý dễ hơn

    # CLAHE tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    # Áp dụng CLAHE để tăng tương phản cục bộ
    variants.append((cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR), "clahe"))
    # Thêm phiên bản ảnh đã tăng tương phản

    # Khử nhiễu
    dn = cv2.fastNlMeansDenoising(clahe, h=8)
    # Khử nhiễu bằng thuật toán Non-local Means
    variants.append((cv2.cvtColor(dn, cv2.COLOR_GRAY2BGR), "denoise"))
    # Thêm phiên bản ảnh đã khử nhiễu

    # Nhị phân hóa adaptive
    at = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 8
    )
    # Nhị phân hóa thích nghi để làm nổi bật ký tự
    variants.append((cv2.cvtColor(at, cv2.COLOR_GRAY2BGR), "adapt_thr"))
    # Thêm phiên bản ảnh nhị phân hóa

    return variants
    # Trả về danh sách các phiên bản ảnh

# ----------------- Xử lý text OCR -----------------
def clean_text_basic(s):
    """Lọc bỏ ký tự thừa, chỉ giữ chữ số, chữ cái, '-' '.'"""
    # Hàm làm sạch kết quả OCR cơ bản

    s = s.strip().upper()
    # Xóa khoảng trắng đầu/cuối và chuyển sang chữ in hoa

    out = []
    # Danh sách ký tự hợp lệ
    for ch in s:
        if ch.isalnum() or ch in "-.":
            out.append(ch)
        # Chỉ giữ chữ cái, chữ số và ký tự cho phép
    return "".join(out)
    # Ghép các ký tự thành chuỗi hoàn chỉnh

def apply_confusion_map(s):
    """Chuyển ký tự dễ nhầm dựa trên bảng CONFUSION_MAP."""
    # Chuẩn hóa các ký tự dễ bị OCR nhận nhầm

    if not s:
        return s
    # Nếu chuỗi rỗng thì trả về luôn

    return "".join(CONFUSION_MAP.get(ch, ch) for ch in s)
    # Ánh xạ từng ký tự qua bảng CONFUSION_MAP

def score_candidate(text, avg_conf):
    """Tính điểm cho một chuỗi OCR dựa trên regex, mã tỉnh, độ dài."""
    # Hàm chấm điểm độ tin cậy cho kết quả OCR

    score = avg_conf
    # Điểm ban đầu dựa trên độ tin cậy trung bình của OCR

    t = text.replace(" ", "")
    # Loại bỏ khoảng trắng trong chuỗi

    # Khớp regex biển số
    for rx in PLATE_REGEXES:
        if rx.match(t):
            score += 0.30
            # Thưởng điểm nếu khớp định dạng biển số Việt Nam
            break

    # Kiểm tra 2 số đầu là mã tỉnh
    if len(t) >= 2 and t[:2].isdigit() and t[:2] in PROVINCE_CODES:
        score += 0.12
        # Thưởng điểm nếu mã tỉnh hợp lệ

    # Phạt chuỗi quá ngắn
    if len(t) < 5:
        score -= 0.18
        # Trừ điểm nếu chuỗi quá ngắn, khả năng sai cao

    # Tăng nhẹ theo độ dài
    score += min(0.05, 0.01 * len(t))
    # Thưởng điểm nhỏ dựa trên số lượng ký tự

    return score
    # Trả về tổng điểm đánh giá

# ----------------- Gom nhóm text nhiều dòng -----------------
def assemble_multiline_segments(ocr_segments, y_threshold_factor=0.6):
    """
    Gom các đoạn text OCR thành dòng (nếu biển số nằm 2 dòng).
    sắp xếp từ trên xuống dưới → trái sang phải.
    """
    # Hàm ghép các đoạn OCR thành một chuỗi biển số hoàn chỉnh

    if not ocr_segments:
        return "", 0.0, []
    # Không có kết quả OCR thì trả về rỗng

    segs = []
    heights = []
    # Danh sách segment và chiều cao từng segment

    for bbox, txt, prob in ocr_segments:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        # Lấy tọa độ các điểm trong bbox

        center_y = float(sum(ys)) / len(ys)
        # Tọa độ Y trung tâm của segment

        left_x = float(min(xs))
        # Tọa độ X bên trái của segment

        height = float(max(ys) - min(ys)) if max(ys) - min(ys) > 0 else 1.0
        # Chiều cao segment

        heights.append(height)
        segs.append({
            "bbox": bbox,
            "text": txt,
            "prob": float(prob) if prob is not None else 0.0,
            "cy": center_y,
            "lx": left_x,
            "h": height
        })
        # Lưu thông tin segment

    median_h = max(1.0, float(np.median(heights)))
    # Lấy chiều cao trung vị để phân tách các dòng

    segs = sorted(segs, key=lambda s: s["cy"])
    # Sắp xếp segment từ trên xuống dưới

    rows = []
    current_row = [segs[0]]
    last_cy = segs[0]["cy"]
    # Khởi tạo dòng đầu tiên

    for s in segs[1:]:
        if abs(s["cy"] - last_cy) > median_h * y_threshold_factor:
            rows.append(current_row)
            # Nếu lệch Y đủ lớn thì tạo dòng mới
            current_row = [s]
            last_cy = s["cy"]
        else:
            current_row.append(s)
            # Nếu cùng dòng thì thêm vào
            last_cy = (last_cy * (len(current_row)-1) + s["cy"]) / len(current_row)
    if current_row:
        rows.append(current_row)
    # Thêm dòng cuối cùng

    row_texts = []
    row_confs = []
    rows_list = []
    # Danh sách text và độ tin cậy từng dòng

    for row in rows:
        row_sorted = sorted(row, key=lambda r: r["lx"])
        # Sắp xếp ký tự trong dòng từ trái sang phải

        texts = [clean_text_basic(r["text"]) for r in row_sorted if clean_text_basic(r["text"])]
        # Làm sạch text OCR

        probs = [r["prob"] for r in row_sorted if r.get("prob") is not None]
        # Lấy độ tin cậy của từng segment

        if not texts:
            continue

        row_text = "".join(texts)
        # Ghép text trong cùng một dòng

        row_texts.append(row_text)
        rows_list.append(row_text)
        row_confs.append(float(np.mean(probs)) if probs else 0.0)
        # Lưu text và độ tin cậy trung bình của dòng

    if not row_texts:
        return "", 0.0, []
    # Nếu không ghép được dòng nào

    assembled = " ".join(row_texts)
    # Ghép các dòng (biển số 2 dòng)
    avg_conf = float(np.mean(row_confs)) if row_confs else 0.0
    # Độ tin cậy trung bình của toàn bộ biển số

    return assembled, avg_conf, rows_list
    # Trả về biển số hoàn chỉnh, độ tin cậy và danh sách từng dòng
# ----------------- OCR nhiều biến thể -----------------
def ocr_ensemble_on_crop(crop):                          # Hàm OCR ensemble cho 1 ảnh crop biển số
    """
    Chạy OCR trên nhiều phiên bản xử lý ảnh + nhiều mức phóng.
    Trả về danh sách text ứng viên, sắp theo điểm số giảm dần.
    """
    candidates = []                                      # Danh sách lưu các ứng viên OCR

    crop_nb = remove_plate_border(crop)                  # Loại bỏ viền/khung dư quanh biển số
    rect = rectify_plate(crop_nb)                        # Căn chỉnh lại biển số cho thẳng (hiệu chỉnh phối cảnh)
    variants = preprocess_variants(rect)                 # Tạo các biến thể tiền xử lý ảnh

    for var_img, desc in variants:                       # Lặp qua từng ảnh biến thể
        for mag in MAG_RATIOS:                            # Lặp qua từng tỉ lệ phóng đại
            img_for_ocr = var_img                         # Ảnh dùng để đưa vào OCR

            # Phóng ảnh nếu quá thấp
            h = img_for_ocr.shape[0]                      # Lấy chiều cao ảnh
            if h < 40:                                    # Nếu ảnh quá nhỏ
                scale = max(1.0, 64.0 / float(h)) * mag   # Tính hệ số phóng lớn hơn để OCR rõ
                img_for_ocr = cv2.resize(                 # Resize ảnh
                    var_img,
                    (int(var_img.shape[1] * scale),       # Chiều rộng mới
                     int(var_img.shape[0] * scale)),      # Chiều cao mới
                    interpolation=cv2.INTER_CUBIC         # Nội suy bicubic cho ảnh rõ nét
                )
            else:
                if mag != 1.0:                            # Nếu chỉ phóng theo mag_ratio
                    img_for_ocr = cv2.resize(             # Resize ảnh
                        var_img,
                        (int(var_img.shape[1] * mag),     # Chiều rộng mới
                         int(var_img.shape[0] * mag)),    # Chiều cao mới
                        interpolation=cv2.INTER_CUBIC     # Nội suy bicubic
                    )

            try:
                res = reader.readtext(                    # Chạy OCR bằng EasyOCR
                    img_for_ocr,
                    detail=1,                             # Trả về bbox + text + confidence
                    allowlist=ALLOWLIST,                  # Chỉ cho phép ký tự hợp lệ
                    paragraph=False,                      # Không gom thành đoạn văn
                    mag_ratio=1.0                         # Không phóng nội bộ OCR
                )
            except Exception:
                res = reader.readtext(                    # OCR fallback nếu có lỗi
                    img_for_ocr,
                    detail=1,
                    allowlist=ALLOWLIST,
                    paragraph=False
                )

            if not res:                                   # Nếu OCR không đọc được gì
                continue                                  # Bỏ qua vòng lặp

            assembled, avg_conf, rows_list = assemble_multiline_segments(
                res,                                      # Kết quả OCR từng segment
                y_threshold_factor=0.6                    # Ngưỡng gom dòng theo trục Y
            )
            if not assembled:                             # Nếu không ghép được text hợp lệ
                continue                                  # Bỏ qua

            raw_score = score_candidate(                  # Tính điểm cho chuỗi OCR gốc
                assembled,
                avg_conf
            )

            mapped = apply_confusion_map(assembled)       # Sửa các ký tự dễ nhầm (O→0, I→1, B→8…)
            mapped_score = score_candidate(               # Tính điểm cho chuỗi đã sửa
                mapped,
                avg_conf
            )

            candidates.append({                           # Lưu ứng viên OCR gốc
                "text": assembled,                        # Chuỗi OCR
                "avg_conf": avg_conf,                     # Độ tin cậy trung bình
                "score": raw_score,                       # Điểm đánh giá
                "variant": desc,                          # Loại tiền xử lý
                "mag": mag,                               # Mức phóng đại
                "rows": rows_list,                        # Text theo từng dòng
                "segments": res                           # Kết quả OCR thô
            })

            if mapped != assembled:                       # Nếu chuỗi sau khi sửa khác chuỗi gốc
                candidates.append({                       # Lưu thêm ứng viên OCR đã sửa
                    "text": mapped,                       # Chuỗi OCR đã sửa
                    "avg_conf": avg_conf,                 # Độ tin cậy trung bình
                    "score": mapped_score,                # Điểm đánh giá
                    "variant": desc + "_mapped",          # Đánh dấu đã qua confusion map
                    "mag": mag,                           # Mức phóng đại
                    "rows": rows_list,                    # Text theo từng dòng
                    "segments": res                       # Kết quả OCR thô
                })

    if not candidates:                                    # Nếu không có ứng viên nào
        return []                                         # Trả về danh sách rỗng

    # Loại trùng, giữ text có điểm cao nhất
    candidates_sorted = sorted(                           # Sắp xếp ứng viên theo điểm giảm dần
        candidates,
        key=lambda x: x["score"],
        reverse=True
    )

    seen = set()                                          # Set lưu text đã xuất hiện
    unique = []                                           # Danh sách kết quả không trùng

    for c in candidates_sorted:                            # Duyệt từng ứng viên đã sắp xếp
        key = c["text"]                                   # Lấy chuỗi OCR làm khóa
        if key in seen:                                   # Nếu chuỗi đã tồn tại
            continue                                      # Bỏ qua
        seen.add(key)                                     # Đánh dấu đã gặp
        unique.append(c)                                  # Thêm ứng viên tốt nhất của chuỗi đó

    return unique                                         # Trả về danh sách OCR cuối cùng

# ----------------- Hàm nhận diện toàn bộ ảnh -----------------
def recognize_license_plate_with_yolo_improved(image_path, debug_topk=3):
    """Chạy YOLO → crop biển số → OCR nhiều lớp → hiển thị kết quả."""
    image = cv2.imread(image_path)                         # Đọc ảnh từ đường dẫn
    if image is None:                                      # Nếu không đọc được ảnh
        print(f"Không đọc được ảnh: {image_path}")         # Thông báo lỗi
        return                                             # Thoát hàm

    print(f"\n--- Đang xử lý ảnh: {os.path.basename(image_path)} ---")  # In tên ảnh đang xử lý

    results = detector(image, conf=0.5, verbose=False)     # Chạy YOLO phát hiện biển số
    found_plate = False                                    # Cờ kiểm tra có nhận diện được biển số hay không

    for result in results:                                 # Duyệt từng kết quả YOLO
        boxes = getattr(result.boxes, "xyxy", None)        # Lấy bounding box dạng xyxy
        if boxes is None:                                  # Nếu không có bbox
            continue                                       # Bỏ qua

        for box in boxes.tolist():                         # Duyệt từng bounding box
            x1, y1, x2, y2 = [int(v) for v in box]         # Ép tọa độ bbox về int
            crop, (xa, ya, xb, yb) = safe_pad_and_crop(    # Crop biển số kèm padding an toàn
                image,
                (x1, y1, x2, y2)
            )
            if crop.size == 0:                             # Nếu crop lỗi / rỗng
                continue                                   # Bỏ qua

            # Phóng biển số quá nhỏ
            if crop.shape[0] < 24:                         # Nếu chiều cao biển số quá nhỏ
                scale = int(np.ceil(32.0 / float(crop.shape[0])))  # Tính hệ số phóng
                crop = cv2.resize(                         # Resize để OCR rõ hơn
                    crop,
                    (crop.shape[1] * scale, crop.shape[0] * scale),
                    interpolation=cv2.INTER_CUBIC
                )

            candidates = ocr_ensemble_on_crop(crop)        # Chạy OCR ensemble trên crop biển số

            if not candidates:                             # Nếu OCR ensemble thất bại
                # Chạy fallback đơn giản
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang grayscale
                clahe = cv2.createCLAHE(                   # Tăng tương phản bằng CLAHE
                    clipLimit=2.0,
                    tileGridSize=(8, 8)
                ).apply(gray)
                try:
                    fallback = reader.readtext(            # OCR fallback bằng EasyOCR
                        clahe,
                        detail=0,                          # Chỉ lấy text
                        allowlist=ALLOWLIST                # Giới hạn ký tự hợp lệ
                    )
                    fallback_text = "".join(               # Ghép các text OCR lại
                        [clean_text_basic(s) for s in fallback]
                    ) if fallback else ""
                except Exception:
                    fallback_text = ""                     # Nếu lỗi OCR thì để rỗng

                if fallback_text:                           # Nếu fallback OCR có kết quả
                    final_text = fallback_text             # Gán text cuối
                    final_score = 0.2                      # Điểm tin cậy thấp cố định
                else:
                    final_text = ""                         # Không có text
                    final_score = 0.0                      # Điểm 0
            else:
                best = candidates[0]                       # Lấy ứng viên OCR tốt nhất
                final_text = best["text"]                  # Text OCR cuối cùng

                # Thử sửa regex (thêm '-' '.' nếu thiếu)
                t_fixed = regex_fix_plate(final_text) if 'regex_fix_plate' in globals() else final_text
                if t_fixed != final_text:                  # Nếu text được sửa định dạng
                    final_text = t_fixed                   # Cập nhật text
                    final_score = best["score"] + 0.12     # Cộng thêm điểm tin cậy
                else:
                    final_score = best["score"]            # Giữ nguyên điểm

            if candidates:                                 # Nếu có danh sách ứng viên OCR
                print(f"Crop bbox [{xa},{ya},{xb},{yb}] - ứng viên OCR:")
                for c in candidates[:debug_topk]:          # In top-K ứng viên
                    print(
                        f"  {c['text']}  conf={c['avg_conf']:.3f} "
                        f"score={c['score']:.3f} ({c['variant']}, mag={c['mag']})"
                    )
                    if c.get("rows"):                      # Nếu biển số nhiều dòng
                        print(f"    dòng: {c['rows']}")

            final_text_clean = clean_text_basic(final_text).replace(" ", "")  # Làm sạch text cuối
            if final_text_clean:                            # Nếu có text hợp lệ
                found_plate = True                          # Đánh dấu đã tìm thấy biển số
                cv2.rectangle(                              # Vẽ bounding box lên ảnh
                    image,
                    (xa, ya),
                    (xb, yb),
                    (0, 255, 0),
                    2
                )
                cv2.putText(                                # Vẽ text biển số lên ảnh
                    image,
                    final_text_clean,
                    (xa, max(ya - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
                print(
                    f"✅ Biển số nhận diện: {final_text_clean} "
                    f"(score ~ {final_score:.3f})"
                )

    if not found_plate:                                     # Nếu không nhận diện được biển số nào
        print("❌ Không tìm thấy biển số hoặc chất lượng ảnh quá kém.")

    # Hiển thị ảnh kết quả
    display_width = 800                                     # Chiều rộng hiển thị
    display_height = 600                                    # Chiều cao hiển thị
    if image is not None and image.shape[0] > 0:            # Kiểm tra ảnh hợp lệ
        resized_image = cv2.resize(                         # Resize ảnh để hiển thị
            image,
            (display_width, display_height)
        )
        cv2.imshow(                                        # Hiển thị ảnh kết quả
            "Kết quả nhận diện (800x600)",
            resized_image
        )
    cv2.waitKey(0)                                          # Chờ phím bất kỳ
    cv2.destroyAllWindows()                                 # Đóng toàn bộ cửa sổ OpenCV

# ----------------- Hàm sửa regex (tự thêm '-' '.') -----------------
def regex_fix_plate(s):
    """Thêm '-' và '.' theo đúng định dạng biển VN nếu thiếu."""
    if not s:                                               # Nếu chuỗi rỗng
        return s
    t = s.replace(" ", "")                                  # Loại bỏ khoảng trắng

    m = re.fullmatch(r"(\d{2}[A-Z])(\d{3})(\d{2})", t)      # Regex dạng 30F44101
    if m:
        return f"{m.group(1)}-{m.group(2)}.{m.group(3)}"   # Trả về dạng 30F-441.01

    m2 = re.fullmatch(r"(\d{2})(\d{5})", t)                 # Regex dạng 1234567
    if m2:
        return f"{m2.group(1)}-{m2.group(2)}"               # Trả về dạng 12-34567

    for rx in PLATE_REGEXES:                                # Kiểm tra các regex hợp lệ khác
        if rx.match(t):
            return t
    return t                                                # Trả về chuỗi ban đầu nếu không khớp

def detect_plate(image_path):
    """
    Hàm dùng cho API.
    Trả về biển số đầu tiên tìm được.
    """

    image = cv2.imread(image_path)
    if image is None:
        return ""

    results = detector(image, conf=0.5, verbose=False)

    for result in results:
        boxes = getattr(result.boxes, "xyxy", None)
        if boxes is None:
            continue

        for box in boxes.tolist():
            x1, y1, x2, y2 = [int(v) for v in box]
            crop, _ = safe_pad_and_crop(image, (x1, y1, x2, y2))

            if crop.size == 0:
                continue

            candidates = ocr_ensemble_on_crop(crop)

            if candidates:
                best = candidates[0]
                text = clean_text_basic(best["text"]).replace(" ", "")
                if text:
                    return text

    return ""

# ----------------- Chạy theo folder images -----------------
# if __name__ == "__main__":
#     image_folder = "images"                                 # Thư mục chứa ảnh đầu vào
#     if not os.path.exists(image_folder):                    # Nếu thư mục không tồn tại
#         print(f"Không tìm thấy folder: {image_folder}")
#     else:
#         imgs = sorted([                                    # Lấy danh sách file ảnh
#             f for f in os.listdir(image_folder)
#             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
#         ])
#         for fn in imgs:                                    # Duyệt từng ảnh
#             path = os.path.join(image_folder, fn)          # Ghép đường dẫn ảnh
#             recognize_license_plate_with_yolo_improved(path)  # Nhận diện biển số
