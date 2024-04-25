import cv2
import requests
import numpy as np
import threading
import json
import time
import logging
from fire_detect import YOLO, detect_fire_and_smoke

COOLDOWN_FRAMES = 60

logging.basicConfig(filename='fire_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def send_fire_alert(video_index, start_time):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    time_taken = time.time() - start_time  # Calcula o tempo decorrido desde a detecção até o envio do alerta
    alert_data = {
        "camera": video_index,
        "timestamp": current_time,
        "alert_message": "Incêndio detectado!",
        "time_taken": time_taken
    }
    json_data = json.dumps(alert_data)
    endpoint_url = "http://localhost:5001/fire-alert"  # Endpoint do servidor cliente consumindo o pacote
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint_url, data=json_data, headers=headers)
    if response.status_code == 200:
        logging.info(f"Alerta de incêndio enviado com sucesso! Câmera: {video_index}, Tempo decorrido: {time_taken} segundos")
    else:
        logging.error("Falha ao enviar alerta de incêndio")

def alert_analyzer(url, video_index, model):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            bytes_data = b''
            c = 0
            cooldown = 0
            for chunk in response.iter_content(chunk_size=4096):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    if cooldown > 0:
                        cooldown -= 1
                    jpg = bytes_data[a:b + 2]
                    bytes_data = bytes_data[b + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    fire_detected, smoke_detected = detect_fire_and_smoke(model, frame)
                    if fire_detected:
                        c+=1
                        start_time = time.time()  # Marca o tempo de início da detecção de incêndio
                        print("Fogo detectado em ", url)
                        if c>=5 and cooldown == 0:
                            thread = threading.Thread(target=send_fire_alert, args=(video_index, start_time))
                            thread.start()
                            # send_fire_alert(video_index, start_time)
                            cooldown = COOLDOWN_FRAMES
                    elif smoke_detected:
                        print("Fumaça detectada em", url)
                    else:
                        c = 0
                    cv2.imshow(url, frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            cv2.destroyWindow(url)
        else:
            logging.error("Falha ao acessar imagem do servidor")
    except Exception as e:
        logging.error(f"Erro: {e}")

if __name__ == '__main__':
    api_urls = [
        {"url": "http://localhost:5000/video_feed?video_index=0", "video_index": 0},
        {"url": "http://localhost:5000/video_feed?video_index=1", "video_index": 1},
        {"url": "http://localhost:5000/video_feed?video_index=2", "video_index": 2},
        {"url": "http://localhost:5000/video_feed?video_index=3", "video_index": 3}
    ]

    model_path = "runs/detect/train3/weights/best.pt"
    model = YOLO(model_path)

    threads = []
    for entry in api_urls:
        thread = threading.Thread(target=alert_analyzer, args=(entry["url"], entry["video_index"], model))
        threads.append(thread)
        thread.start()
