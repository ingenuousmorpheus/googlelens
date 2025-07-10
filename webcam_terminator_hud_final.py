
import cv2
from ultralytics import YOLO
import requests
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import time

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# Attempt to load custom font
font_path = "ocr-a.ttf" if os.path.exists("ocr-a.ttf") else "arial.ttf"
font_size = 14
try:
    font = ImageFont.truetype(font_path, font_size)
except IOError:
    font = ImageFont.load_default()

def draw_terminator_overlay(frame, box, label_text, flicker):
    x1, y1, x2, y2 = map(int, box)

    # Draw red rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Convert to PIL for better text rendering
    pil_img = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_img)

    # Flashing "LOCKED" text
    if flicker:
        draw.text((x1, y1 - 20), "[LOCKED]", font=font, fill=(255, 0, 0))

    # Draw analysis text
    lines = label_text.split("\n")
    overlay_x, overlay_y = x2 + 10, y1
    for i, line in enumerate(lines):
        draw.text((overlay_x, overlay_y + i * 16), line, font=font, fill=(255, 64, 64))

    return np.array(pil_img)

def analyze_person(image_crop):
    # Dummy analysis — customize with LLaVA integration as needed
    return (
        "[ANALYSIS]\n"
        "- SUBJECT LOCKED\n"
        "- WEIGHT: ~180 lbs\n"
        "- RACE: HISPANIC\n"
        "- SHIRT: WHITE TANK, DARK BLAZER\n"
        "- HAIR: LONG BLACK"
    )

flicker = True
last_toggle = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]

    # Flip flicker every 0.5s
    if time.time() - last_toggle > 0.5:
        flicker = not flicker
        last_toggle = time.time()

    for result in results.boxes:
        if int(result.cls[0]) == 0:  # Person
            x1, y1, x2, y2 = result.xyxy[0]
            cropped = frame[int(y1):int(y2), int(x1):int(x2)]

            description = analyze_person(cropped)
            frame = draw_terminator_overlay(frame, (x1, y1, x2, y2), description, flicker)

    cv2.imshow("T-800 Vision", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
