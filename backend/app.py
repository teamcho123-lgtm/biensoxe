import streamlit as st
import cv2
import numpy as np
import easyocr
import re
from ultralytics import YOLO

# ----------------- Cấu hình -----------------
DEFAULT_YOLO_WEIGHTS = "best.pt" 
ALLOWLIST = "0123456789ABCDEFGHIJKLMNPRSTUVXYZ-."
MAG_RATIOS = [1.0, 1.5, 2.0]
PAD_RATIO = 0.05

# Chữ cái hợp lệ trên biển số VN (loại bỏ I, O, Q, J, W vì dễ nhầm)
VN_LETTERS = set("ABCDEFGHKLMNPRSTUVXYZ")

# Regex định dạng biển số VN
PLATE_PATTERNS = [
    re.compile(r"^\d{2}[A-Z]-\d{3}\.\d{2}$"),  # 30F-441.01
    re.compile(r"^\d{2}-\d{5}$"),              # 12-34567
    re.compile(r"^\d{2}[A-Z]-\d{4}$"),         # 30F-1234
    re.compile(r"^\d{2}[A-Z]\d{5}$"),          # 30F12345
]

PROVINCE_CODES = {f"{i:02d}" for i in range(1, 100)}

# ----------------- Load Models -----------------
@st.cache_resource
def load_models(weights_path, use_gpu):
    try:
        detector = YOLO(weights_path)
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        return detector, reader, None
    except Exception as e:
        return None, None, str(e)

# ----------------- Xử lý ảnh -----------------
def safe_crop(img, xyxy, pad_ratio=PAD_RATIO):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
    xa = max(0, x1 - pad_x)
    ya = max(0, y1 - pad_y)
    xb = min(w - 1, x2 + pad_x)
    yb = min(h - 1, y2 + pad_y)
    return img[ya:yb, xa:xb].copy()

def enhance_image(img):
    """Cải thiện chất lượng ảnh cho OCR"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tăng tương phản
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Khử nhiễu nhẹ
    denoised = cv2.fastNlMeansDenoising(enhanced, h=7)
    
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def resize_for_ocr(img, min_height=60):
    """Resize ảnh để phù hợp với OCR"""
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Giới hạn kích thước tối đa
        if new_w > 600:
            scale = 600 / w
            new_w = 600
            new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return img

# ----------------- Xử lý text -----------------
def clean_text(text):
    """Làm sạch text OCR"""
    text = text.strip().upper().replace(" ", "")
    return "".join(c for c in text if c.isalnum() or c in "-.")

def fix_char_confusion(text, position, expected_type):
    """Sửa nhầm lẫn ký tự dựa trên vị trí và loại kỳ vọng"""
    if not text or position >= len(text):
        return text
    
    char = text[position]
    
    if expected_type == "digit":
        # Vị trí cần số: sửa O->0, I->1, S->5, Z->2, B->8
        fixes = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", 
                 "S": "5", "Z": "2", "B": "8"}
        return fixes.get(char, char)
    elif expected_type == "letter":
        # Vị trí cần chữ: sửa 0->O, 1->I, 5->S, 2->Z, 8->B
        fixes = {"0": "O", "1": "I", "5": "S", "2": "Z", "8": "B"}
        fixed = fixes.get(char, char)
        # Chỉ giữ chữ cái hợp lệ trên biển số VN
        if fixed in VN_LETTERS:
            return fixed
        return char
    return char

def normalize_plate(text):
    """Chuẩn hóa biển số theo định dạng VN"""
    text = clean_text(text)
    if not text:
        return ""
    
    # Loại bỏ dấu phân cách để phân tích
    compact = text.replace("-", "").replace(".", "")
    
    # Mẫu: 37M00079 -> 37M-000.79 (biển số 2 dòng)
    # Tìm pattern: 2 số + 1 chữ + 5 số
    match = re.match(r"^(\d{2})([A-Z])(\d{5})$", compact)
    if match:
        province = match.group(1)
        letter = match.group(2)
        numbers = match.group(3)
        
        # Sửa 2 số đầu (mã tỉnh) - chỉ sửa nếu thực sự cần
        province_fixed = "".join(fix_char_confusion(province, i, "digit") for i in range(len(province)))
        if not province_fixed.isdigit() or len(province_fixed) != 2:
            province_fixed = province
        
        # Sửa chữ cái - chỉ sửa nếu không phải chữ hợp lệ
        if letter not in VN_LETTERS:
            letter = fix_char_confusion(letter, 0, "letter")
        if letter not in VN_LETTERS:
            return text  # Không thể sửa được
        
        # Sửa phần số - chỉ sửa các ký tự không phải số
        numbers_fixed = ""
        for i, char in enumerate(numbers):
            if char.isdigit():
                numbers_fixed += char
            else:
                fixed = fix_char_confusion(numbers, i, "digit")
                if fixed.isdigit():
                    numbers_fixed += fixed
                else:
                    numbers_fixed += char  # Giữ nguyên nếu không sửa được
        
        if len(numbers_fixed) == 5 and numbers_fixed.isdigit():
            return f"{province_fixed}{letter}-{numbers_fixed[:3]}.{numbers_fixed[3:]}"
    
    # Mẫu: 30F44101 -> 30F-441.01 (biển số 1 dòng)
    match2 = re.match(r"^(\d{2})([A-Z0-9])(\d{3,5})$", compact)
    if match2:
        province = match2.group(1)
        letter = match2.group(2)
        numbers = match2.group(3)
        
        province_fixed = "".join(fix_char_confusion(province, i, "digit") for i in range(len(province)))
        if not province_fixed.isdigit() or len(province_fixed) != 2:
            return text
        
        if letter not in VN_LETTERS:
            letter = fix_char_confusion(letter, 0, "letter")
        if letter not in VN_LETTERS:
            return text
        
        numbers_fixed = "".join(fix_char_confusion(numbers, i, "digit") for i in range(len(numbers)))
        if not numbers_fixed.isdigit():
            return text
        
        if len(numbers_fixed) == 5:
            return f"{province_fixed}{letter}-{numbers_fixed[:3]}.{numbers_fixed[3:]}"
        elif len(numbers_fixed) == 4:
            return f"{province_fixed}{letter}-{numbers_fixed}"
        elif len(numbers_fixed) == 3:
            return f"{province_fixed}{letter}-{numbers_fixed}"
    
    # Mẫu: 12-34567 (xe máy)
    match3 = re.match(r"^(\d{2})(\d{5})$", compact)
    if match3:
        province = "".join(fix_char_confusion(match3.group(1), i, "digit") for i in range(2))
        numbers = "".join(fix_char_confusion(match3.group(2), i, "digit") for i in range(5))
        if province.isdigit() and numbers.isdigit():
            return f"{province}-{numbers}"
    
    return text

def score_plate(text, confidence):
    """Tính điểm cho kết quả OCR"""
    score = confidence
    text_clean = clean_text(text)
    
    # Thưởng điểm nếu khớp định dạng
    for pattern in PLATE_PATTERNS:
        if pattern.match(text_clean):
            score += 0.4
            break
    
    # Thưởng điểm nếu mã tỉnh hợp lệ
    if len(text_clean) >= 2 and text_clean[:2].isdigit():
        if text_clean[:2] in PROVINCE_CODES:
            score += 0.2
    
    # Phạt nếu quá ngắn
    if len(text_clean) < 5:
        score -= 0.3
    
    return score

# ----------------- OCR -----------------
def read_plate_ocr(img, reader):
    """Đọc biển số từ ảnh (đơn giản, ưu tiên giữ đủ ký tự)."""
    # 1) Tiền xử lý & resize 1 lần
    enhanced = enhance_image(img)
    img_ocr = resize_for_ocr(enhanced)

    try:
        results = reader.readtext(
            img_ocr,
            detail=1,
            allowlist=ALLOWLIST,
            paragraph=False
        )
    except Exception:
        return None, 0.0

    if not results:
        return None, 0.0

    # 2) Lọc & sắp xếp segment theo vị trí (trên→dưới, trái→phải)
    segments = []
    for bbox, txt, conf in results:
        cleaned = clean_text(txt)
        if not cleaned:
            continue
        # bbox: 4 điểm, lấy tâm để sắp xếp
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = float(sum(xs)) / len(xs)
        cy = float(sum(ys)) / len(ys)
        segments.append((cy, cx, cleaned, conf if conf is not None else 0.0))

    if not segments:
        return None, 0.0

    # sort by y then x để giữ đúng thứ tự 2 dòng biển số
    segments.sort(key=lambda s: (s[0], s[1]))

    combined_text = "".join(seg[2] for seg in segments)
    avg_conf = float(np.mean([seg[3] for seg in segments])) if segments else 0.0

    normalized = normalize_plate(combined_text)
    score = score_plate(normalized, avg_conf)

    # Nếu chuẩn hoá làm chuỗi tệ hơn (ngắn hơn nhiều), giữ bản gốc
    if len(clean_text(normalized)) + 1 < len(clean_text(combined_text)):
        normalized = combined_text

    return normalized, avg_conf

# ----------------- Streamlit UI -----------------
st.set_page_config(
    page_title="Nhận diện biển số xe",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Hệ Thống Nhận Diện Biển Số Xe")
st.markdown("---")

st.sidebar.header("⚙️ Cấu hình")
gpu_option = st.sidebar.checkbox("Sử dụng GPU", value=False)
conf_threshold = st.sidebar.slider("Ngưỡng YOLO", 0.1, 1.0, 0.5)

with st.spinner("Đang tải mô hình..."):
    detector, reader, error_msg = load_models(DEFAULT_YOLO_WEIGHTS, gpu_option)

if error_msg:
    st.error(f"❌ Lỗi: {error_msg}")
    st.stop()

uploaded_file = st.file_uploader("Tải lên ảnh xe", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    st.subheader("🖼️ Ảnh Đầu Vào")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    if st.button("🔍 Nhận diện"):
        with st.spinner("Đang xử lý..."):
            results = detector(image, conf=conf_threshold, verbose=False)
            
            found_plates = []
            output_image = image.copy()
            
            for result in results:
                boxes = getattr(result.boxes, "xyxy", None)
                if boxes is None:
                    continue
                
                for box in boxes.tolist():
                    x1, y1, x2, y2 = [int(v) for v in box]
                    crop = safe_crop(image, (x1, y1, x2, y2))
                    
                    if crop.size == 0:
                        continue
                    
                    # Đọc biển số
                    plate_text, confidence = read_plate_ocr(crop, reader)
                    
                    if plate_text and len(plate_text) >= 5:
                        found_plates.append({
                            "text": plate_text,
                            "confidence": confidence,
                            "crop": crop,
                            "bbox": (x1, y1, x2, y2)
                        })
                        
                        # Vẽ lên ảnh
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(output_image, plate_text, (x1, max(y1 - 10, 0)),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Kết Quả")
                st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            with col2:
                st.subheader("📄 Chi Tiết")
                if found_plates:
                    for i, plate in enumerate(found_plates):
                        st.success(f"**Biển số #{i+1}: {plate['text']}**")
                        st.caption(f"Độ tin cậy: {plate['confidence']:.2%}")
                        st.image(cv2.cvtColor(plate['crop'], cv2.COLOR_BGR2RGB), 
                                width=300, caption="Ảnh crop")
                else:
                    st.warning("Không tìm thấy biển số nào.")
