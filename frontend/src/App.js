import './App.css';
import { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from "react-webcam";

function App() {
  // Thêm state quản lý Dark Mode (mặc định là false - màu hồng)
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [preview, setPreview] = useState(null);

  const [history, setHistory] = useState([]);
  const [result, setResult] = useState("");
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);

  const webcamRef = useRef(null);
  const runningRef = useRef(false);
  const detectingRef = useRef(true);
  const modeRef = useRef("camera");

  const toggleCamera = () => {
    setIsCameraOn(prev => !prev);
  };

  // 🎥 REALTIME LOOP (Logic giữ nguyên)
  const captureLoop = useCallback(async () => {

    if (!isCameraOn) return;
    if (!webcamRef.current) return;
    if (runningRef.current) return;
    if (!detectingRef.current) return;
    if (modeRef.current !== "camera") return; // 🚨 QUAN TRỌNG

    runningRef.current = true;

    try {

      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        runningRef.current = false;
        setTimeout(() => captureLoop(), 800);
        return;
      }

      const blob = await fetch(imageSrc).then(res => res.blob());

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch("http://127.0.0.1:8000/detect", {
        method: "POST",
        body: formData,
        cache: "no-store"   // 🚨 CHỐNG CACHE
      });

      const data = await response.json();

      if (data.plate) {
        setResult(data.plate);
      }

    } catch (err) {
      console.error(err);
    }

    runningRef.current = false;

    setTimeout(() => captureLoop(), 800);

  }, [isCameraOn]);

  useEffect(() => {
    if (isCameraOn) captureLoop();
  }, [isCameraOn, captureLoop]);

  // 📸 CHỤP CAMERA (Logic giữ nguyên)
  const saveImage = async () => {
    modeRef.current = "capture";
    detectingRef.current = false;

    if (!webcamRef.current) return;
    setIsCapturing(true);
    detectingRef.current = false;
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) {
        setIsCapturing(false);
        detectingRef.current = true;
        return;
      }
      const blob = await fetch(imageSrc).then(res => res.blob());
      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetch("http://127.0.0.1:8000/detect", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!data.plate) {
        setIsCapturing(false);
        detectingRef.current = true;
        return;
      }
      const newItem = {
        plate: data.plate,
        image: imageSrc,
        bbox: data.bbox,
        time: new Date().toLocaleTimeString('vi-VN', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        })
      };
      setResult(data.plate);
      setHistory(prev => [newItem, ...prev]);
    } catch (err) {
      console.error(err);
    }
    detectingRef.current = true;
    setIsCapturing(false);

    modeRef.current = "camera";
    detectingRef.current = true;
  };

  // 📂 UPLOAD ẢNH TỪ MÁY (Logic giữ nguyên)
  const uploadImage = async (e) => {
    modeRef.current = "upload";
    detectingRef.current = false;

    const file = e.target.files[0];
    if (!file) return;
    setIsCapturing(true);
    detectingRef.current = false;
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch("http://127.0.0.1:8000/detect", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (!data.plate) {
        alert("Không tìm thấy biển số trong ảnh này!");
        setIsCapturing(false);
        detectingRef.current = true;
        e.target.value = null;
        return;
      }
      const imageURL = URL.createObjectURL(file);
      const newItem = {
        plate: data.plate,
        image: imageURL,
        time: new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      };
      setResult(data.plate);
      setHistory(prev => [newItem, ...prev]);

      modeRef.current = "camera";
      detectingRef.current = true;
    } catch (err) {
      console.error(err);
    }
    e.target.value = null;
    detectingRef.current = true;
    setIsCapturing(false);
  };

  // Áp dụng class dark-theme vào thẻ bao ngoài cùng
  return (
    <div className={`App ${isDarkMode ? 'dark-theme' : ''}`}>
      <div className="dashboard-container">

        {/* HEADER */}
        <header className="app-header">
          <div className="header-title">
            {/* Đổi icon theo theme */}
            <span className="logo-icon">{isDarkMode ? '🦇' : '🎀'}</span>
            <h1>LPR System</h1>
          </div>

          {/* Khu vực nút bấm góc phải */}
          <div className="header-actions">
            <button
              className="theme-toggle"
              onClick={() => setIsDarkMode(!isDarkMode)}
            >
              {isDarkMode ? '🌸 Đổi màu Hồng' : '🌙 Đổi màu Dark'}
            </button>

            <div className={`status-badge ${isCameraOn ? 'online' : 'offline'}`}>
              {isCameraOn ? 'Camera Active' : 'Camera Off'}
            </div>
          </div>
        </header>

        {/* MAIN CONTENT */}
        <div className="main-content">
          <div className="control-panel">

            <div className="action-card">
              <h3>BẢNG ĐIỀU KHIỂN</h3>
              <div className="button-group">
                <button
                  className={`btn ${isCameraOn ? 'btn-danger' : 'btn-primary'}`}
                  onClick={toggleCamera}
                >
                  {isCameraOn ? "⏹ DỪNG CAMERA" : "▶️ BẬT CAMERA"}
                </button>

                <div className="upload-wrapper">
                  <label htmlFor="file-upload" className={`btn btn-upload ${isCapturing ? 'disabled' : ''}`}>
                    📂 TẢI ẢNH LÊN
                  </label>
                  <input
                    id="file-upload"
                    type="file"
                    accept="image/*"
                    onChange={uploadImage}
                    disabled={isCapturing}
                    className="hidden-input"
                  />
                </div>
              </div>
            </div>

            <div className="camera-card">
              <div className="camera-view-container">
                {isCameraOn ? (
                  <>
                    <Webcam
                      className="webcam-video"
                      audio={false}
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      screenshotQuality={0.9}
                      width={640}
                      height={360}
                      videoConstraints={{
                        width: 640,
                        height: 360,
                        facingMode: "environment"
                      }}
                    />
                    <div className="scan-overlay"></div>
                  </>
                ) : (
                  <div className="camera-placeholder">
                    <span>Màn hình chờ</span>
                  </div>
                )}
              </div>

              {isCameraOn && (
                <button
                  className="btn btn-capture-now"
                  onClick={saveImage}
                  disabled={isCapturing}
                >
                  {isCapturing ? "ĐANG XỬ LÝ..." : "📸 CHỤP MÀN HÌNH NÀY"}
                </button>
              )}
            </div>

            <div className="result-card">
              <h3>KẾT QUẢ HIỆN TẠI</h3>
              <div className="plate-display">
                {result ? result : <span className="plate-placeholder">-- --</span>}
              </div>
            </div>

          </div>

          <div className="history-panel">
            <div className="history-header">
              <h3>LỊCH SỬ NHẬN DIỆN</h3>
              <span className="history-count">{history.length}</span>
            </div>

            <div className="history-list">
              {history.length === 0 ? (
                <div className="empty-state">Chưa có dữ liệu</div>
              ) : (
                history.map((item, index) => (
                  <div
                    className="history-item"
                    key={index}
                    onClick={() => setPreview(item)}
                  >
                    <div className="history-img-wrapper">
                      <img src={item.image} alt="plate" />
                    </div>
                    <div className="history-info">
                      <p className="history-plate">{item.plate}</p>
                      <p className="history-time">{item.time}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
      {preview && (
        <div className="preview-modal" onClick={() => setPreview(null)}>
          <div className="preview-box" onClick={(e) => e.stopPropagation()}>

            <div className="preview-image-wrapper">
              <img src={preview.image} alt="" />

              {preview.bbox && (
                <div
                  className="plate-box"
                  style={{
                    left: preview.bbox[0],
                    top: preview.bbox[1],
                    width: preview.bbox[2] - preview.bbox[0],
                    height: preview.bbox[3] - preview.bbox[1]
                  }}
                ></div>
              )}
            </div>

            <h2>{preview.plate}</h2>
            <p>{preview.time}</p>

            <button onClick={() => setPreview(null)}>
              Đóng
            </button>

          </div>
        </div>
      )}
    </div>
  );
}

export default App;