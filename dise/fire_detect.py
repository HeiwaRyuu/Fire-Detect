import cv2
import numpy as np
import torch
from mss import mss
from ultralytics import YOLO
import time

def get_center_screen_coordinates(width=608, height=608):
    with mss() as sct:
        monitor = sct.monitors[1]
        ## CENTER
        center_left = (monitor["width"] - width) // 2
        center_top = (monitor["height"] - height) // 2

        return center_left, center_top, width, height

def letterbox(image, expected_size):
    """Resize image to expected size, maintaining aspect ratio with padding."""
    ih, iw = image.shape[:2]
    eh, ew = expected_size, expected_size

    # SCALE
    scale = min(ew/iw, eh/ih)
    nw, nh = int(iw * scale), int(ih * scale)
    image_resized = cv2.resize(image, (nw, nh))

    # PADD
    # top = 20 #(eh - nh) // 2
    # bottom = 0 #eh - nh - top
    # left = 0 #(ew - nw) // 2
    # right = 0 #ew - nw - left

    # APP PADD
    # image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    
    return image_resized


def draw_bounding_box(frame, box, class_name, confidence):
    # COORDS
    x1, y1, x2, y2 = [int(coord) for coord in box]

    # LABEL BG
    label = f"{class_name} {confidence:.2f}"
    (text_width, text_height), baseline  = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    text_start_x = x1
    if y1>text_height+baseline:
        text_start_y = y1-text_height
    else:
        text_start_y = y1+text_height
    
    # BOUNDING BOX RECT
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # LABEL RECT
    cv2.rectangle(frame, (text_start_x, text_start_y-text_height), (text_start_x + text_width + baseline, text_start_y + text_height), (0, 0, 255), cv2.FILLED)
    # LABEL
    cv2.putText(frame, label, (text_start_x+baseline, text_start_y+baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def detect_fire_and_smoke(model, frame, time_window=1000, detection_threshold=1):
    results = model.predict(frame, verbose=False)
    fire_detected = False
    smoke_detected = False
    result = results[0]
    confidence_threshold = 0.001

    # Lista para armazenar os tempos de detecção de fogo
    fire_detection_times = []

    for i in range(len(result.boxes.data)):
        cls_id = int(result.boxes.cls[i].item())
        conf = result.boxes.conf[i].item()
        if conf > confidence_threshold:
            bbox = result.boxes.xyxy[i].tolist()  # [x1, y1, x2, y2]
            class_name = result.names[cls_id].lower()
            draw_bounding_box(frame, bbox, class_name, conf)
            if class_name == 'fire':
                # Registrar o tempo da detecção de fogo
                fire_detection_times.append(time.time())
                # Remover os tempos de detecção que estão fora da janela de tempo
                fire_detection_times = [t for t in fire_detection_times if time.time() - t <= time_window]
                # Se o número de detecções dentro da janela de tempo atingir o limite, considera-se fogo detectado
                if len(fire_detection_times) >= detection_threshold:
                    fire_detected = True
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            elif class_name == 'smoke':
                smoke_detected = True
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    return fire_detected, smoke_detected



def capture_webcam(model):
    # Initialize the webcam (0 is the default webcam, adjust if necessary)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            # If frame is read correctly, ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) ## CONVERTING TO RGB
            # frame = letterbox(frame, model_input_size) ## ONLY IF I CHANGE CAPTURE REGION

            is_fire, is_smoke = detect_fire_and_smoke(model, frame)
            if is_fire:
                print("Fire detected!")
            if is_smoke:
                print("Smoke detected!")

            cv2.imshow('WebCam', cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)) ## DISPLAY

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def screen_capture(model):
    model_input_size = 640
    left, top, width, height = get_center_screen_coordinates(width=model_input_size, height=model_input_size)
    
    # CAPTURE REGION
    mon = {
        "top": top,
        "left": left,
        "width": width,
        "height": height
    }

    with mss() as sct:
        try:
            while True:
                # SCREEN CAPTURE
                sct_img = sct.grab(mon)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) ## CONVERTING TO BGR
                # frame = letterbox(frame, model_input_size) ## ONLY IF I CHANGE CAPTURE REGION

                is_fire, is_smoke = detect_fire_and_smoke(model, frame)
                if is_fire:
                    print("Fire detected!")
                if is_smoke:
                    print("Smoke detected!")

                cv2.imshow('Screen', cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)) ## DISPLAY

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyAllWindows()

def main():
    ## USING GPU IF AVAILABLE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path="runs\\detect\\train3\\weights\\best.pt"
    model = YOLO(model_path).to(device)

    capture = "webcam"

    if capture=="screen":
        screen_capture(model)
    if capture=="webcam":
        capture_webcam(model)

    

if __name__ == "__main__":
    main()