import cv2
import numpy as np

# Cargar el modelo de YOLOv4-tiny
net = cv2.dnn.readNetFromDarknet("DNN/yolov4-tiny.cfg", "DNN/yolov4-tiny.weights")

# Guardar los nombres de las etiquetas en un arreglo
classes = []
with open("DNN/classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, img = cap.read()
    if not ret:
        print("Error al capturar el frame")
        break

    # Obtener el alto y ancho de la imagen
    ht, wt, _ = img.shape

    # Crear un blob a partir de la imagen
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    # Configurar la entrada de la red
    net.setInput(blob)

    # Obtener los nombres de las últimas capas
    last_layer = net.getUnconnectedOutLayersNames()

    # Ejecutar la red
    layer_out = net.forward(last_layer)

    # Inicializar listas para las cajas, las confidencias y los IDs de clase
    boxes = []
    confidences = []
    class_ids = []

    # Procesar las salidas de cada capa
    for output in layer_out:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                center_x = int(detection[0] * wt)
                center_y = int(detection[1] * ht)
                w = int(detection[2] * wt)
                h = int(detection[3] * ht)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar Non-Maximum Suppression para eliminar duplicados
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    # Dibujar las cajas y etiquetas en la imagen
    if len(indexes) > 0:
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            # Dibujar la caja delimitadora
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # Poner el texto de la etiqueta y la confianza
            cv2.putText(img, f'{label} {confidence}', (x, y + 20), font, 2, (0, 0, 0), 2)

    # Mostrar la imagen con las detecciones
    cv2.imshow('Imagen', img)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows(1)
