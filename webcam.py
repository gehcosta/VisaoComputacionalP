import numpy as np
import cv2
import time

# Carrega o modelo pré-treinado para detecção de objetos
net = cv2.dnn.readNet('frozen_inference_graph.pb', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')

# Inicializa a webcam
cap = cv2.VideoCapture(0)

while True:
    # Captura um frame da webcam
    ret, frame = cap.read()

    # Obtém as dimensões do frame
    height, width, _ = frame.shape

    # Cria um blob a partir do frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Define a entrada da rede neural
    net.setInput(blob)

    # Realiza a detecção de objetos
    detections = net.forward()

    # Loop sobre as detecções
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confiança mínima para detecção
            class_id = int(detections[0, 0, i, 1])
            label = f'Objeto {class_id}'
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostra o frame com as detecções
    cv2.imshow('Detecção de Objetos', frame)

    # Verifica se o usuário pressionou a tecla 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a webcam e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()