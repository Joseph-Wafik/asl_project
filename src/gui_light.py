"""
Lightweight ASL GUI with:
- Advanced real-time detection (MediaPipe if available)
- Single image prediction
- Folder prediction
- Grad-CAM visualization
"""

# Shehab Hossam

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import time
import json

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# Optional MediaPipe
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

# Project imports
from config import IMG_SIZE, SAVED_MODELS_DIR, DATA_DIR, NUM_CLASSES
from model_builder import ModelBuilder
from data_preprocessing import get_preprocessing_settings


# --------------------------
# Hand detector (trimmed)
# --------------------------
class RealTimeHandDetector:
    def __init__(self, use_mediapipe=True):
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.hand_history = []
        self.prediction_history = []
        self.last_hand_time = time.time()
        self.hand_lost_threshold = 1.0
        self.current_results = None
        self.hands = None
        # track how many consecutive detections we have
        self.detection_streak = 0
        self.min_streak_for_prediction = 2
        self.min_roi_size = 60  # pixels
        self.padding_scale = 0.5
        if self.use_mediapipe:
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.6,
                model_complexity=1,
            )
        else:
            self.mp_hands = None
            self.mp_drawing = None

    def detect(self, frame):
        # MediaPipe path
        if self.use_mediapipe and self.hands is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_results = self.hands.process(frame_rgb)
            if self.current_results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                # pick the largest hand by bbox area
                best_bbox = None
                best_area = 0
                for lm in self.current_results.multi_hand_landmarks:
                    xs = [p.x * w for p in lm.landmark]
                    ys = [p.y * h for p in lm.landmark]
                    x1, x2 = int(min(xs)), int(max(xs))
                    y1, y2 = int(min(ys)), int(max(ys))
                    area = max(1, (x2 - x1) * (y2 - y1))
                    if area > best_area:
                        best_area = area
                        best_bbox = (x1, y1, x2, y2)
                if best_bbox is None:
                    self.detection_streak = 0
                    return None
                x1, y1, x2, y2 = best_bbox
                width = x2 - x1
                height = y2 - y1
                if min(width, height) < self.min_roi_size:
                    self.detection_streak = 0
                    return None
                pad = int(max(width, height) * self.padding_scale)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                size = max(width, height) + pad
                x1, y1 = max(0, cx - size // 2), max(0, cy - size // 2)
                x2, y2 = min(w, cx + size // 2), min(h, cy + size // 2)
                self.last_hand_time = time.time()
                self.hand_history.append(((x1, y1, x2, y2), time.time()))
                if len(self.hand_history) > 8:
                    self.hand_history.pop(0)
                self.detection_streak += 1
                return x1, y1, x2, y2
        # No detection if MediaPipe not available or no hand
        self.current_results = None
        self.detection_streak = 0
        return None

    def get_smoothed(self):
        if not self.hand_history:
            return None
        now = time.time()
        wx1 = wy1 = wx2 = wy2 = wsum = 0
        for (x1, y1, x2, y2), ts in self.hand_history:
            w = max(0, 1.0 - (now - ts) / 2.0)
            wx1 += x1 * w; wy1 += y1 * w
            wx2 += x2 * w; wy2 += y2 * w
            wsum += w
        if wsum == 0:
            return None
        return (int(wx1 / wsum), int(wy1 / wsum), int(wx2 / wsum), int(wy2 / wsum))

    def is_hand_present(self):
        recent = (time.time() - self.last_hand_time) < self.hand_lost_threshold
        streak_ok = self.detection_streak >= self.min_streak_for_prediction
        return recent and streak_ok

    def draw(self, frame, bbox, prediction=None, confidence=0, show_box=True):
        if self.use_mediapipe and self.current_results and self.current_results.multi_hand_landmarks:
            for hand_landmarks in self.current_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                )
        if show_box and bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if prediction:
                text = f"{prediction} {confidence:.1f}%"
                cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def cleanup(self):
        if self.hands:
            self.hands.close()


# --------------------------
# Grad-CAM (concise)
# --------------------------
class FixedGradCAM:
    def __init__(self, model, last_conv_name=None):
        self.model = model
        self.layer_name = last_conv_name or self._find_last_conv()

    def _find_last_conv(self):
        conv_layers = [l.name for l in self.model.layers if 'conv' in l.name.lower()]
        return conv_layers[-1] if conv_layers else self.model.layers[-1].name

    def compute(self, img_array, class_idx=None):
        target_layer = self.model.get_layer(self.layer_name)
        grad_model = tf.keras.models.Model(self.model.inputs, [target_layer.output, self.model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            if class_idx is None:
                class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        grads = tape.gradient(loss, conv_out)[0]
        if grads is None:
            return np.zeros(conv_out[0].shape[:2])
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(weights * conv_out[0], axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam.numpy() if hasattr(cam, "numpy") else cam
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# --------------------------
# Prediction Smoothing
# --------------------------
class PredictionSmoother:
    def __init__(self, num_classes, alpha=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.current_probs = np.zeros(num_classes)
        self.is_first = True

    def update(self, new_probs):
        if self.is_first:
            self.current_probs = new_probs
            self.is_first = False
        else:
            self.current_probs = self.alpha * new_probs + (1 - self.alpha) * self.current_probs
        return self.current_probs

    def reset(self):
        self.current_probs = np.zeros(self.num_classes)
        self.is_first = True


# --------------------------
# GUI
# --------------------------
class ASLLightGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Lite GUI - Advanced Real-Time + Grad-CAM")

        # State
        self.model = None
        self.class_indices = None
        self.gradcam = None
        self.camera_active = False
        self.cap = None
        self.hand_detector = RealTimeHandDetector()
        self.current_image = None
        self.gradcam_active = False
        
        # Advanced Prediction
        self.smoother = None
        self.prediction_threshold = 0.7
        self.last_frame_time = time.time()
        self.fps = 0

        # UI
        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self.root, pady=8)
        top.pack(fill="x")

        tk.Label(top, text="Model:").pack(side="left")
        self.model_var = tk.StringVar(value="EfficientNetB0")
        ttk.Combobox(top, textvariable=self.model_var,
                     values=["EfficientNetB0", "ResNet50", "InceptionV3"],
                     state="readonly", width=18).pack(side="left", padx=5)

        tk.Button(top, text="Load Weights", command=self.load_weights, bg="#3498DB", fg="white").pack(side="left", padx=5)
        self.cam_btn = tk.Button(top, text="Start Camera", command=self.toggle_camera, bg="#27AE60", fg="white")
        self.cam_btn.pack(side="left", padx=5)
        tk.Button(top, text="Upload Image", command=self.load_image, bg="#9B59B6", fg="white").pack(side="left", padx=5)
        tk.Button(top, text="Upload Folder", command=self.load_folder, bg="#E67E22", fg="white").pack(side="left", padx=5)
        self.gradcam_btn = tk.Button(top, text="Grad-CAM OFF", command=self.run_gradcam, bg="#E74C3C", fg="white")
        self.gradcam_btn.pack(side="left", padx=5)

        # Canvas
        self.canvas = tk.Canvas(self.root, width=800, height=500, bg="black")
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)

        # Info labels
        info = tk.Frame(self.root)
        info.pack(fill="x", padx=10)
        self.pred_label = tk.Label(info, text="Prediction: --", font=("Arial", 16, "bold"), fg="blue")
        self.pred_label.pack(side="left", padx=5)
        self.conf_label = tk.Label(info, text="Confidence: --", font=("Arial", 12))
        self.conf_label.pack(side="left", padx=5)
        self.fps_label = tk.Label(info, text="FPS: --", font=("Arial", 10))
        self.fps_label.pack(side="right", padx=5)
        
        # Top-3 predictions label (for single image only)
        top3_frame = tk.Frame(self.root)
        top3_frame.pack(fill="x", padx=10, pady=(5, 0))
        self.top3_label = tk.Label(top3_frame, text="", font=("Arial", 10), fg="#95A5A6", justify="left")
        self.top3_label.pack(side="left", padx=5)

        # Log
        log_frame = tk.Frame(self.root)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, height=8, bg="#1C2833", fg="#ECF0F1")
        self.log.pack(side="left", fill="both", expand=True)
        tk.Scrollbar(log_frame, command=self.log.yview).pack(side="right", fill="y")
        self.log.config(yscrollcommand=lambda *args: self.log.yview(*args))

    # ---------- Model ----------
    def load_weights(self):
        path = filedialog.askopenfilename(
            title="Select weights",
            filetypes=[("H5 Weights", "*.h5")],
            initialdir=str(SAVED_MODELS_DIR),
        )
        if not path:
            return
        try:
            builder = ModelBuilder(self.model_var.get(), num_classes=NUM_CLASSES)
            self.model = builder.build_model()
            self.model.load_weights(path)
            self.gradcam = FixedGradCAM(self.model)
            self._load_class_indices()
            
            # Init smoother
            self.smoother = PredictionSmoother(NUM_CLASSES, alpha=0.7) # Higher alpha = faster response
            
            messagebox.showinfo("Loaded", f"Loaded weights: {Path(path).name}")
            self.log.insert("end", f"✅ Loaded weights {path}\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.log.insert("end", f"❌ Load error: {e}\n")

    def _load_class_indices(self):
        idx_path = DATA_DIR / "class_indices.json"
        if idx_path.exists():
            with open(idx_path, "r", encoding="utf-8") as f:
                indices = json.load(f)
            self.class_indices = {int(v): k for k, v in indices.items()}
        else:
            self.class_indices = {i: chr(65 + i) for i in range(NUM_CLASSES)}

    # ---------- Prediction helpers ----------
    def preprocess(self, img):
        if img is None:
            return None
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        elif img_resized.shape[2] == 4:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGRA2RGB)
        else:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        prep_fn, rescale = get_preprocessing_settings(self.model_var.get())
        if prep_fn:
            arr = prep_fn(img_resized.astype("float32"))
        else:
            arr = img_resized.astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)

    def predict(self, img, smooth=False):
        if self.model is None:
            return None, 0, []
        batch = self.preprocess(img)
        if batch is None:
            return None, 0, []
            
        probs = self.model.predict(batch, verbose=0)[0]
        
        # Apply smoothing if requested and available
        if smooth and self.smoother:
            probs = self.smoother.update(probs)
            
        top_idx = np.argsort(probs)[::-1][:3]
        classes = [self.class_indices.get(i, f"Class_{i}") for i in top_idx]
        confs = [probs[i] for i in top_idx] # Keep as float 0-1
        
        return classes, confs, probs # Return full probs for visualization

    # ---------- Camera ----------
    def toggle_camera(self):
        if self.camera_active:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load model weights first!")
            return
            
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            self.camera_active = True
            if self.smoother:
                self.smoother.reset()
            self.cam_btn.config(text="Stop Camera", bg="#E74C3C")
            self._update_camera()
            self.log.insert("end", "📸 Camera started\n")
        except Exception as e:
            messagebox.showerror("Camera", str(e))

    def stop_camera(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.cam_btn.config(text="Start Camera", bg="#27AE60")
        self.log.insert("end", "⏹️ Camera stopped\n")
        self.pred_label.config(text="Prediction: --")
        self.conf_label.config(text="Confidence: --")
        self.fps_label.config(text="FPS: --")
        self.top3_label.config(text="")

    def _update_camera(self):
        if not self.camera_active:
            return
            
        # FPS Calculation
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1 / dt)
        self.fps_label.config(text=f"FPS: {int(self.fps)}")

        ret, frame = self.cap.read()
        if not ret:
            self.stop_camera()
            return

        bbox = self.hand_detector.detect(frame)
        smoothed_box = self.hand_detector.get_smoothed()
        hand_visible = smoothed_box is not None and self.hand_detector.is_hand_present()

        top_class = None
        top_conf = 0
        raw_probs = None
        
        if hand_visible:
            x1, y1, x2, y2 = smoothed_box
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                # Predict with smoothing
                classes, confs, raw_probs = self.predict(roi, smooth=True)
                top_class = classes[0]
                top_conf = confs[0]
                
                # keep a copy for Grad-CAM
                self.current_image = frame.copy()

        frame_disp = frame.copy()
        
        # Draw HUD
        if hand_visible and self.hand_detector.current_results is not None:
             # Draw Skeleton
            self.hand_detector.draw(frame_disp, smoothed_box, None, 0, show_box=False)
            
            # Draw Advanced Info
            if top_class and top_conf > self.prediction_threshold:
                 color = (0, 255, 0)
                 text = f"{top_class} ({top_conf*100:.1f}%)"
            else:
                 color = (0, 165, 255) # Orange for uncertain
                 text = "..."
                 
            # Display Main Prediction on Frame
            cv2.putText(frame_disp, text, (smoothed_box[0], max(30, smoothed_box[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Update Labels
        if hand_visible and top_class and top_conf > self.prediction_threshold:
            self.pred_label.config(text=f"Prediction: {top_class}", fg="#2ECC71") # Green
            self.conf_label.config(text=f"Confidence: {top_conf*100:.1f}%")
            self.top3_label.config(text="")  # Clear top-3 for real-time
        elif hand_visible:
            self.pred_label.config(text="Prediction: ...", fg="#F1C40F") # Yellow
            self.conf_label.config(text=f"Low Confidence ({top_conf*100:.1f}%)")
            self.top3_label.config(text="")  # Clear top-3 for real-time
        else:
            self.pred_label.config(text="Prediction: --", fg="blue")
            self.pred_label.config(text="Prediction: --")
            self.conf_label.config(text="Confidence: --")
            self.top3_label.config(text="")  # Clear top-3 when no hand

        # Convert to RGB before drawing on Tk canvas
        frame_disp_rgb = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)
        self._draw_on_canvas(frame_disp_rgb)
        self.root.after(10, self._update_camera)

    def _draw_stats(self, frame, probs):
        """Draws a Top-3 probability bar chart on the frame"""
        h, w = frame.shape[:2]
        
        # Get top 3
        top_idx = np.argsort(probs)[::-1][:3]
        
        bar_x = 20
        bar_y = 60
        bar_h = 20
        bar_w = 150
        gap = 10
        
        # Background for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 40), (250, 40 + 3 * (bar_h + gap) + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        for i, idx in enumerate(top_idx):
            class_name = self.class_indices.get(idx, f"C{idx}")
            prob = probs[idx]
            
            # Label
            text = f"{prob*100:.0f}% {class_name}"
            cv2.putText(frame, text, (bar_x, bar_y + i*(bar_h+gap) + bar_h - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Bar
            filled_w = int(bar_w * prob)
            cv2.rectangle(frame, (bar_x + 80, bar_y + i*(bar_h+gap)), 
                          (bar_x + 80 + filled_w, bar_y + i*(bar_h+gap) + bar_h), 
                          (0, 255, 0) if i==0 else (0, 255, 255), -1)

    # ---------- Image / Folder (Updated) ----------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select image",
                                          filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot load image")
            return
        self.current_image = img.copy()
        classes, confs, probs = self.predict(img, smooth=False)
        pred, conf = classes[0], confs[0]
        
        # Display image without any overlays
        self._draw_on_canvas(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if pred:
            # Update main prediction labels
            self.pred_label.config(text=f"Prediction: {pred}", fg="black")
            self.conf_label.config(text=f"Confidence: {conf*100:.1f}%")
            
            # Update top-3 predictions label
            top3_text = "Top 3:\n"
            for i, (cls, cnf) in enumerate(zip(classes[:3], confs[:3]), 1):
                top3_text += f"  {i}. {cls}: {cnf*100:.1f}%\n"
            self.top3_label.config(text=top3_text)
            
            # Log to console
            log_text = f"Image {Path(path).name}:\n"
            for i, (cls, cnf) in enumerate(zip(classes[:3], confs[:3]), 1):
                log_text += f"  {i}. {cls}: {cnf*100:.1f}%\n"
            self.log.insert("end", log_text)
        else:
            self.pred_label.config(text="Prediction: --")
            self.conf_label.config(text="Confidence: --")
            self.top3_label.config(text="")

    def load_folder(self):
        folder = filedialog.askdirectory(title="Select folder")
        if not folder:
            return
        # Ask user for ground truth mode
        dialog = tk.Toplevel(self.root)
        dialog.title("Ground Truth Source")
        dialog.geometry("320x180")
        tk.Label(dialog, text="Use which source for correct label?", font=("Arial", 11, "bold")).pack(pady=10)
        mode_var = tk.StringVar(value="folder")
        tk.Radiobutton(dialog, text="Folder name", variable=mode_var, value="folder").pack(anchor="w", padx=20)
        tk.Radiobutton(dialog, text="File name", variable=mode_var, value="filename").pack(anchor="w", padx=20)

        def start():
            dialog.destroy()
            threading.Thread(target=self._predict_folder, args=(folder, mode_var.get()), daemon=True).start()

        tk.Button(dialog, text="Start", command=start, bg="#2ECC71", fg="white").pack(pady=10)
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)

    def _predict_folder(self, folder, mode="folder"):
        # Clear top-3 label for folder prediction
        self.root.after(0, lambda: self.top3_label.config(text=""))
        
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for ext in exts:
            paths.extend(Path(folder).rglob(ext))
        if not paths:
            self.log.insert("end", "No images found in folder.\n")
            return
        correct = 0
        total = 0
        wrong = 0
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            classes, confs, _ = self.predict(img, smooth=False)
            pred = classes[0]
            conf = confs[0]
            
            # determine ground truth
            if mode == "folder":
                gt = p.parent.name
            else:
                stem = p.stem
                gt = stem.split("_")[0]
            total += 1
            is_correct = pred is not None and pred.upper() == str(gt).upper()
            if is_correct:
                correct += 1
            else:
                wrong += 1
            self.log.insert("end", f"{p.name}: pred={pred or '--'} ({conf*100:.1f}%) | gt={gt} | {'✅' if is_correct else '❌'}\n")
            self.log.see("end")
        acc = (correct / total * 100) if total else 0.0
        self.log.insert("end", f"Done. {total} images. Correct {correct}, Wrong {wrong}, Accuracy {acc:.1f}%\n")
        messagebox.showinfo("Folder Test Complete", f"Images: {total}\nCorrect: {correct}\nWrong: {wrong}\nAccuracy: {acc:.1f}%")

    # ---------- Grad-CAM ----------
    def run_gradcam(self):
        # Toggle on/off
        self.gradcam_active = not self.gradcam_active
        if self.gradcam_active:
            self.gradcam_btn.config(text="Grad-CAM ON", bg="#27AE60")
        else:
            self.gradcam_btn.config(text="Grad-CAM OFF", bg="#E74C3C")
            # show plain image when turning off
            if self.current_image is not None:
                self._draw_on_canvas(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
            return

        if self.model is None:
            messagebox.showwarning("Grad-CAM", "Load model first.")
            self.gradcam_active = False
            self.gradcam_btn.config(text="Grad-CAM OFF", bg="#E74C3C")
            return
        if self.current_image is None:
            messagebox.showwarning("Grad-CAM", "Load an image or show your hand to the camera first.")
            self.gradcam_active = False
            self.gradcam_btn.config(text="Grad-CAM OFF", bg="#E74C3C")
            return
        if self.gradcam is None:
            self.gradcam = FixedGradCAM(self.model)
        try:
            batch = self.preprocess(self.current_image)
            heatmap = self.gradcam.compute(batch)
            heatmap = cv2.resize(heatmap, (self.current_image.shape[1], self.current_image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            base = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(base, 0.5, heatmap, 0.5, 0)
            self._draw_on_canvas(overlay)
            self.log.insert("end", "Grad-CAM generated for current image.\n")
        except Exception as e:
            messagebox.showerror("Grad-CAM Error", str(e))
            self.log.insert("end", f"Grad-CAM error: {e}\n")
            self.gradcam_active = False
            self.gradcam_btn.config(text="Grad-CAM OFF", bg="#E74C3C")

    # ---------- Display ----------
    def _draw_on_canvas(self, frame_rgb):
        if frame_rgb is None or frame_rgb.size == 0:
            return
        if frame_rgb.shape[2] == 3:
            disp = frame_rgb
        else:
            disp = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        h, w = disp.shape[:2]
        c_w = max(10, self.canvas.winfo_width())
        c_h = max(10, self.canvas.winfo_height())
        aspect = w / h
        if c_w / c_h > aspect:
            nh = c_h
            nw = int(c_h * aspect)
        else:
            nw = c_w
            nh = int(c_w / aspect)
        disp = cv2.resize(disp, (nw, nh))
        img_pil = Image.fromarray(disp)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(c_w // 2, c_h // 2, image=img_tk, anchor="center")
        self.canvas.image = img_tk

    # ---------- Cleanup ----------
    def cleanup(self):
        self.stop_camera()
        self.hand_detector.cleanup()


if __name__ == "__main__":
    root = tk.Tk()
    app = ASLLightGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

