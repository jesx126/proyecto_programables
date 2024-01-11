import cv2


# Arquitectura del modelo pre entrenado
prototxt = "MobileNetSSD_deploy.prototxt.txt"
# Pesos contenidos del modelo
model = "MobileNetSSD_deploy.caffemodel"
# Todas las etiquetas que el haarcascade soporta para mostrar en los recuadros delimitadores
classes = {0:"background", 1:"aeroplane", 2:"bisicleta",
          3:"pajaro", 4:"barco",
          5:"bote", 6:"bus",
          7:"carro", 8:"gato",
          9:"silla", 10:"vaca",
          11:"comedor", 12:"perro",
          13:"caballo", 14:"motocicleta",
          15:"persona", 16:"maceta",
          17:"oveja", 18:"sofa",
          19:"tren", 20:"pantalla"}

# uso de open cv para carcar el modelo
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# captura de la imagen mediante el servidor 
cap = cv2.VideoCapture("http://192.168.137.21" + ":81/stream")

while True:
     ret, frame, = cap.read()
     if ret == False:
          break

     height, width, _ = frame.shape
     frame_resized = cv2.resize(frame, (300, 300))

     # creación del blob
     blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))
     #print("blob.shape:", blob.shape)

     #Algoritmo de edtecciónes y predicciónes basado en el estándar de opencv
     net.setInput(blob)
     #detección hacia delante
     detections = net.forward()

     for detection in detections[0][0]:
          #print(detection)

          if detection[2] > 0.45:
               label = classes[detection[1]]
               #print("Label:", label)
               box = detection[3:7] * [width, height, width, height]
               x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])

               cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
               cv2.putText(frame, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (255, 0, 0), 2)
               cv2.putText(frame, label, (x_start, y_start - 25), 1, 1.5, (0, 255, 255), 2)


     cv2.imshow("Frame", frame)
     if cv2.waitKey(1) & 0xFF == 27:
          break
cap.release()
cv2.destroyAllWindows()