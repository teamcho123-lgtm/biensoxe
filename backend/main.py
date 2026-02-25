from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from datetime import datetime
import uvicorn
from yolo_lpr import detect_plate

app = FastAPI()

# Cho React gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# DETECT BIỂN SỐ
# =============================
@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    # Lưu ảnh tạm
    with open("temp.jpg", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Đọc ảnh để lấy size thật
    img = cv2.imread("temp.jpg")
    h, w = img.shape[:2]

    # YOLO detect
    result = detect_plate("temp.jpg")

    if isinstance(result, tuple):
        plate, bbox = result
    else:
        plate = result
        bbox = None

    return {
        "plate": plate,
        "bbox": bbox,
        "img_width": w,
        "img_height": h
    }


# =============================
# SAVE ẢNH
# =============================
@app.post("/save")
async def save_image(file: UploadFile = File(...)):

    if not os.path.exists("saved"):
        os.makedirs("saved")

    filename = f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join("saved", filename)

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {
        "message": "saved",
        "filename": filename
    }


# =============================
# RUN SERVER
# =============================
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)