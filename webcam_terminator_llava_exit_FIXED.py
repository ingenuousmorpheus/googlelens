
import cv2
from ultralytics import YOLO
import requests
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import time
import tkinter as tk
import threading
import io
import base64

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Load font
font_path = "ocr-a.ttf" if os.path.exists("ocr-a.ttf") else "arial.ttf"
try:
    font = ImageFont.truetype(font_path, 14)
except IOError:
    font = ImageFont.load_default()

# Analyze person using LLaVA via LM Studio
def analyze_person(image_crop):
    try:
        pil_image = Image.fromarray(image_crop)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        payload = {
            "model": "llava-v1.5-7b",
            "messages": [
                {"role": "system", "content": "You are a visual AI that describes people like a Terminator scan."},
                {"role": "user", "content": "Describe this person's weight, race, and clothing."}
            ],
            "temperature": 0.2,
            "max_tokens": 300,
            "stream": False,
            "images": [img_base64]
        }

        response = requests.post("http://127.0.0.1:1234/v1/chat/completions", json=payload)
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR]\n{e}"

def draw_terminator_overlay(frame, box, label_text, flicker):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Place "[LOCKED]" above the head or inside box if space is limited
    lock_y = y1 - 20 if y1 > 30 else y1 + 5
    if flicker:
        draw.text((x1, lock_y), "[LOCKED]", font=font, fill=(255, 0, 0))

    lines = label_text.split("\n")

    # Align description inside top-left corner of bounding box
    overlay_x, overlay_y = x1 + 5, y1 + 20
    for i, line in enumerate(lines):
        draw.text((overlay_x, overlay_y + i * 16), line, font=font, fill=(255, 0, 0))

    return np.array(pil_img)

class TerminatorVisionApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("T-800 Control Panel")
        self.running = True
        self.analysis_cache = {}

        exit_btn = tk.Button(self.root, text="Exit", command=self.shutdown)
        exit_btn.pack(padx=10, pady=5)

        threading.Thread(target=self.run_vision, daemon=True).start()
        self.root.mainloop()

    def shutdown(self):
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def run_vision(self):
        flicker = True
        last_toggle = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            results = model(frame, verbose=False)[0]

            if time.time() - last_toggle > 0.5:
                flicker = not flicker
                last_toggle = time.time()

            for result in results.boxes:
                if int(result.cls[0]) == 0:
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    crop_key = f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}"

                    if crop_key not in self.analysis_cache:
                        cropped = frame[int(y1):int(y2), int(x1):int(x2)]
                        description = analyze_person(cropped)
                        self.analysis_cache[crop_key] = description
                    else:
                        description = self.analysis_cache[crop_key]

                    frame = draw_terminator_overlay(frame, (x1, y1, x2, y2), description, flicker)

            cv2.imshow("T-800 Vision HUD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.shutdown()

# Run the app
if __name__ == "__main__":
    TerminatorVisionApp()