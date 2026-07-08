from torchvision import transforms
from ultralytics import YOLO
import cv2
import torch
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from knowledge_graph import KnowledgeGraph
from rclpy.node import Node
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from yolo_msgs.msg import DetectionArray
from nav_msgs.msg import Odometry
from shapely.geometry import Point, Polygon
from std_msgs.msg import Empty

from message_filters import Subscriber, ApproximateTimeSynchronizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class CrearGrafo(Node):
    def __init__(self):
        super().__init__("crear_grafo_yolo")

        # Parametros iniciales
        self.frame_count = 0

        self.next_node_id = 0

        self.umbral_tiempo = 3

        self.umbral_distancia = 0.5

        self.min_frames_consecutivos = 2

        self.score_threshold = 0.72

        # Cargar modelo
        self.pesos = models.ResNet18_Weights.IMAGENET1K_V1
        self.model_reid = models.resnet18(weights=self.pesos).to(device)

        # Modificar red para que devuelva directamente el embedding
        self.model_reid.classifier = torch.nn.Identity()

        # Modo evaluacion
        self.model_reid.eval()

        self.frame_mostrar = None

        self.transformacion = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Cargar estancias del mapa
        estancias = {
            "cocina": [[4.76, 3.0], [9.35, 3.11], [8.8, -4.8], [5.0, -5.5]],
            "salon": [[-2.3, 5.4], [4.7, 5.5], [5.0, -5.5], [-2.3, -5.5]],
            "habitacion": [[-9.25, -1.5], [-2.5, -1.5], [-2.5, 3.0], [-9.25, 3.08]],
            "gimnasio": [[-2.5, -1.6], [-2.3, -5.5], [-9.25, -5.5], [-9.25, -1.5]],
        }

        # Obtener las estancias en forma de poligonos
        self.poligonos_estancias = {
            nombre: Polygon(coords) for nombre, coords in estancias.items()
        }

        # Historial global de entidades detectadas
        self.node_history = defaultdict(
            lambda: {
                "cls_id": None,
                "class_instance_id": None,
                "track_id": None,
                "frames": [],
                "position_3d": [],
                "raw_position_3d": [0.0, 0.0, 0.0],
                "embeddings": [],
                "in_graph": False,
                "score": 0,
                "diagonal": 0,
            }
        )

        self.next_instance_per_class = defaultdict(int)

        self.bridge = CvBridge()

        # Grafo de conocimiento
        self.graph = KnowledgeGraph.get_instance()

        for est in estancias.keys():
            self.graph.create_node(name=est, type_="Room")

        # Definir un color por clase
        self.model_yolo = YOLO("yolo26n.pt")
        self.colores_clase = {}
        for cls_id in self.model_yolo.names.keys():
            self.colores_clase[int(cls_id)] = (
                (37 * cls_id) % 256,
                (97 * cls_id) % 256,
                (173 * cls_id) % 256,
            )

        self.subscription = Subscriber(self, Image, "/head_front_camera/image")

        self.detection_3d_subscription = Subscriber(
            self, DetectionArray, "/yolo/detections_3d"
        )

        self.subscription_odom = Subscriber(
            self, Odometry, "/mobile_base_controller/odom"
        )

        queue_size = 10
        max_delay = 0.01

        # Sincronizar subscriebers
        self.time_sync = ApproximateTimeSynchronizer(
            [self.subscription, self.detection_3d_subscription, self.subscription_odom],
            queue_size,
            max_delay,
        )

        self.time_sync.registerCallback(self.detections_3d_callback)

        self.get_logger().info("Sistema cargado. Listo para empezar")
        self.get_logger().info("Presione enter para comenzar.")
        input()

    def hacer_cuadrado_con_padding(self, img_cv2):
        """
        Anhade padding negro para convertir la imagen en un cuadrado perfecto
        sin deformar el contenido original.
        """
        h, w = img_cv2.shape[:2]
        if h == w:
            return img_cv2

        diferencia = abs(h - w)
        top, bottom, left, right = 0, 0, 0, 0

        if h > w:
            left = diferencia // 2
            right = diferencia - left
        else:
            top = diferencia // 2
            bottom = diferencia - top

        img_cuadrada = cv2.copyMakeBorder(
            img_cv2, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        return img_cuadrada

    def extraer_embeddings(self, img):
        """
        Extrae los embeddings de una imagen cuadrada
        """

        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return None

        img_cuadrada = self.hacer_cuadrado_con_padding(img)

        img_rgb = cv2.cvtColor(img_cuadrada, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        img_tensor = img_tensor.unsqueeze(0).to(device)

        img_tensor = self.transformacion(img_tensor)

        with torch.no_grad():
            embeddings = self.model_reid(img_tensor)

            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def obtener_estancia(self, x, y):
        """
        Recibe una coordenada X, Y y comprueba en que estancia se encuentra el objeto
        """

        punto = Point(x, y)
        for nombre_habitacion, poligono in self.poligonos_estancias.items():
            if poligono.contains(punto):
                return nombre_habitacion
        return "desconocido"

    def detections_3d_callback(self, img, msg, odom):
        """
        Filtra y organiza los mensajes sincronizados recibidos de los topics
        """

        frame = self.bridge.imgmsg_to_cv2(img, "bgr8")
        self.frame_count += 1

        pose_robot = [odom.pose.pose.position.x, odom.pose.pose.position.y]

        ids = {}

        for det in msg.detections:

            cls_id = int(det.class_id)
            track_id = int(det.id) if det.id != "" else -1

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size.x
            h = det.bbox.size.y
            score = det.score

            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2

            box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32, device=device)

            if det.bbox3d is not None:
                c3d = det.bbox3d.center.position
                x3d = c3d.x
                y3d = c3d.y
                z3d = c3d.z

                s3d = det.bbox3d.size
                size_x = s3d.x
                size_y = s3d.y
                size_z = s3d.z
                t_objeto = np.sqrt(size_x**2 + size_y**2 + size_z**2)
            else:
                x3d, y3d, z3d, t_objeto = 0.0, 0.0, 0.0, 0.0

            if det.mask.data:
                pts = np.array(
                    [[int(p.x), int(p.y)] for p in det.mask.data], dtype=np.int32
                )

                # Crear mascara binaria
                h_frame, w_frame = frame.shape[:2]
                mask_bin = np.zeros((h_frame, w_frame), dtype=np.uint8)
                cv2.fillPoly(mask_bin, [pts], 255)

            else:
                mask_bin = None

            if cls_id not in ids:
                ids[cls_id] = []

            # Almacena informacion necesaria de cada objeto
            ids[cls_id].append(
                {
                    "box": box,
                    "track_id": track_id,
                    "x3d": x3d,
                    "y3d": y3d,
                    "z3d": z3d,
                    "name": det.class_name,
                    "score": score,
                    "diagonal": t_objeto,
                    "mask": mask_bin,
                }
            )

        # Comprobar si ya se ha detectado algun objeto con anterioridad
        nodos = self.fusionar_objetos(ids, frame, self.frame_count, pose_robot)

        self.extraer_relaciones(nodos, frame)

    def comprobar_posicion(
        self, pose_propuesto, pose_consolidado, pose_robot, tamano_objeto
    ):
        """
        Comprobar si dos objetos estan cerca dadas sus posiciones 3D
        """

        Xp, Yp, Zp = pose_propuesto
        Xc, Yc, Zc = pose_consolidado

        distancia = np.sqrt((Xp - Xc) ** 2 + (Yp - Yc) ** 2 + (Zp - Zc) ** 2)

        distancia_objeto_robot = np.sqrt(
            (Xp - pose_robot[0]) ** 2 + (Yp - pose_robot[1]) ** 2
        )

        umbral_adaptativo = self.umbral_distancia_adaptativo(
            distancia_objeto_robot, tamano_objeto
        )

        return distancia <= umbral_adaptativo, distancia, umbral_adaptativo

    def comprobar_embeddings(self, embeddings_propuestos, node_id):
        """
        Determina si dos embeddings se parecen lo suficiente como para determinar que pertenecen al mismo objeto
        """

        embs = self.node_history[node_id]["embeddings"]
        if not embs:
            return False, 0.0

        if len(embs) >= 3:
            last = embs[-3:]
            stacked = torch.cat(last, dim=0)
            embeddings_consolidados = stacked.mean(dim=0)
        else:
            embeddings_consolidados = embs[-1]

        similitud = F.cosine_similarity(embeddings_propuestos, embeddings_consolidados)

        puntuacion = similitud.item()

        self.get_logger().info(f"Puntuacion de similitud: {puntuacion}")

        return puntuacion > self.score_threshold, puntuacion

    def umbral_distancia_adaptativo(self, distancia, tamano_objeto):
        """
        Calcula un umbral de distancia dinámico basado en la profundidad.
        Cuanto mayor es z (metros), mayor es la tolerancia.
        """

        if tamano_objeto > 1.3:
            # Parámetros ajustables:
            umbral_base = (
                self.umbral_distancia
            )  # umbral mínimo para objetos muy cercanos
            factor_escala = 0.05  # incremento por metro de profundidad
            factor_tamano = 0.25
            umbral_maximo = 3.0  # límite superior para evitar umbrales desmedidos

            umbral = (
                umbral_base + factor_escala * distancia + factor_tamano * tamano_objeto
            )

        else:
            # Parámetros ajustables:
            umbral_base = (
                self.umbral_distancia
            )  # umbral mínimo para objetos muy cercanos
            factor_escala = 0.05  # incremento por metro de profundidad
            factor_tamano = 0.13
            umbral_maximo = 2.0  # límite superior para evitar umbrales desmedidos

            umbral = (
                umbral_base + factor_escala * distancia + factor_tamano * tamano_objeto
            )

        self.get_logger().info(f"Umbral adaptativo: {min(umbral, umbral_maximo)}")
        return min(umbral, umbral_maximo)

    def fusionar_objetos(self, propuestas, frame, frame_count, pose_robot):
        """
        Función para la re-identificacion de objetos

        Parámetros:
            propuestas: diccionario con objetos detectados (puede haber varios objetos por cada clave)
                {clave (id de la clase detectada por el modelo): box(tensor de 4 elementos), track_id (id del tracker si obtiene alguno si no es -1),
                x3d (X de pose 3d), y3d (Y de pose 3d), z3d (Z de pose 3d), name (nombre de la clase)}

            frame: frame actual

            frame_count: numero de frame actual

            pose_robot: posicion del robot actual en el frame del mapa [x, y]


        Return:
            nodos_activos: nodos que se tienen en cuenta tras el preprocesado
        """

        # Conjunto de nodos que ya han sido re-identificados
        obj_consolildados_asignados = set()

        # Nodos presentes en el frame actual
        nodos_activos = {}

        self.get_logger().info("EMPIEZA")

        for id_clase, detecciones in propuestas.items():

            # Recorrer las detecciones de cada clase
            for det in detecciones:
                box = det["box"]
                track_id = det["track_id"]
                X = det["x3d"]
                Y = det["y3d"]
                Z = det["z3d"]
                pose_propuesto = [X, Y, Z]
                name = det["name"]
                score = det["score"]
                t_objeto = det["diagonal"]
                mask = det["mask"]

                self.get_logger().info(f"Nombre: {name}")
                self.get_logger().info(
                    f"Posicion del objeto observado: {pose_propuesto}"
                )

                x1, y1, x2, y2 = map(int, box.cpu().numpy())

                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, min(x1, w_frame - 1))
                y1 = max(0, min(y1, h_frame - 1))
                x2 = max(0, min(x2, w_frame))
                y2 = max(0, min(y2, h_frame))

                distancia_objeto_robot = np.sqrt(
                    (X - pose_robot[0]) ** 2 + (Y - pose_robot[1]) ** 2
                )

                # Solo se tienen en cuenta objetos detectados a menos de 5 metros por temas de precision
                if distancia_objeto_robot >= 5.0:
                    continue

                if mask is not None:
                    frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
                    crop_propuesta = frame_masked[y1:y2, x1:x2]

                    if crop_propuesta.max() == 0:
                        crop_propuesta = frame[y1:y2, x1:x2]

                else:
                    crop_propuesta = frame[y1:y2, x1:x2]

                embeddings_propuestos = self.extraer_embeddings(crop_propuesta)

                best_node_id = -1
                best_score = -1000.0

                best_misma_posicion = False
                best_mismo_track_id = False
                best_embedding = False
                best_distancia_posicion = float("inf")
                best_umbral = 0.0

                # Recorre los objetos ya presentes en el grafo
                for node_id, datos_nodo in self.node_history.items():

                    misma_posicion = False
                    mismo_track_id = False
                    mismo_embedding = False

                    # El objeto consolidado ya se a asignado a otro propuesto
                    if node_id in obj_consolildados_asignados:
                        continue

                    # No es un objeto de la misma clase que el objeto propuesto
                    if id_clase != datos_nodo["cls_id"]:
                        continue

                    pose_consolidado = datos_nodo["position_3d"]
                    t_objeto_consolidado = datos_nodo["diagonal"]
                    self.get_logger().info(
                        f"Tamaño del objeto consolidado: {t_objeto_consolidado}"
                    )

                    # Comprobrar si la posicion del objeto propuesto esta cerca del objeto consolidado
                    misma_posicion, distancia_posicion, umbral_adaptativo = (
                        self.comprobar_posicion(
                            pose_propuesto,
                            pose_consolidado,
                            pose_robot,
                            t_objeto_consolidado,
                        )
                    )

                    self.get_logger().info(
                        f"Distancia de {name} a {datos_nodo["class_instance_id"]}: {distancia_posicion}"
                    )
                    self.get_logger().info(
                        f"Distancia del robot al objeto: {distancia_objeto_robot}"
                    )

                    # Comprobar si se le ha asignado el mismo track de video en el ultimo frame
                    mismo_track_id = (
                        track_id != -1 and track_id == datos_nodo["track_id"]
                    )

                    # Comprobar si los embeddings son similares
                    if (
                        embeddings_propuestos is not None
                        and len(datos_nodo["embeddings"]) > 0
                    ):
                        mismo_embedding, score_embedding_propuesto = (
                            self.comprobar_embeddings(embeddings_propuestos, node_id)
                        )

                        if mismo_embedding:
                            score_embedding = score_embedding_propuesto
                        else:
                            score_embedding = 0
                    else:
                        score_embedding = 0

                    if misma_posicion and distancia_posicion <= 0.45:
                        best_node_id = node_id
                        best_misma_posicion = misma_posicion
                        best_distancia_posicion = distancia_posicion
                        best_mismo_track_id = mismo_track_id
                        break

                    # Si es posible que sea el mismo objeto, se actualiza la mejor puntuacion
                    if (
                        misma_posicion
                        or mismo_track_id
                        or (
                            not misma_posicion
                            and mismo_embedding
                            and distancia_posicion <= umbral_adaptativo * 1.50
                        )
                    ):
                        if -distancia_posicion * 0.3 + score_embedding > best_score:
                            best_score = -distancia_posicion * 0.3 + score_embedding
                            best_node_id = node_id
                            best_distancia_posicion = distancia_posicion
                            best_misma_posicion = misma_posicion
                            best_mismo_track_id = mismo_track_id
                            best_umbral = umbral_adaptativo
                            best_embedding = mismo_embedding

                    self.get_logger().info(
                        f"Score: {-distancia_posicion * 0.3 + score_embedding}"
                    )

                # Si se ha decido que el objeto ya existe y coincide con otro existente en el grafo
                # Se actualiza su información
                if (
                    best_misma_posicion
                    or best_mismo_track_id
                    or (
                        not best_misma_posicion
                        and best_embedding
                        and best_distancia_posicion <= best_umbral * 1.50
                    )
                ):
                    self.get_logger().info("Objeto re-identificado")

                    self.node_history[best_node_id]["frames"].append(frame_count)

                    num_frames_historia = len(self.node_history[best_node_id]["frames"])

                    # Entre mas se ve el objeto, menos se permite que cambie su centro
                    alpha = max(0.05, 0.3 * (1.0 / num_frames_historia))

                    old_x, old_y, old_z = self.node_history[best_node_id]["position_3d"]

                    # Calcular nueva posición suavizada
                    new_x = (1 - alpha) * old_x + alpha * X
                    new_y = (1 - alpha) * old_y + alpha * Y
                    new_z = (1 - alpha) * old_z + alpha * Z

                    self.node_history[best_node_id]["position_3d"] = [
                        new_x,
                        new_y,
                        new_z,
                    ]

                    self.node_history[best_node_id]["raw_position_3d"] = [X, Y, Z]

                    # Si el score es mayor se actualiza
                    if score > self.node_history[best_node_id]["score"]:
                        self.node_history[best_node_id]["score"] = score

                        if embeddings_propuestos is not None and (
                            best_misma_posicion or best_mismo_track_id
                        ):
                            self.node_history[best_node_id]["embeddings"].append(
                                embeddings_propuestos
                            )

                    # Si se ha producido un movimiento del objeto y se ha desplazado demasiado (esto no deberia ocurrir nunca)
                    if not best_misma_posicion and best_distancia_posicion > 2.0:
                        self.node_history[best_node_id]["position_3d"] = [X, Y, Z]

                    # Se actualiza el tamanho
                    if (
                        t_objeto > self.node_history[best_node_id]["diagonal"]
                        and score >= self.node_history[best_node_id]["score"]
                    ):
                        self.node_history[best_node_id]["diagonal"] = t_objeto

                    obj_consolildados_asignados.add(best_node_id)

                    habitacion = self.obtener_estancia(X, Y)

                    # Crear nodo si se ha detectado en un minimo de frames
                    frames = self.node_history[best_node_id]["frames"]
                    if len(frames) >= self.min_frames_consecutivos:
                        if (
                            frames[-1] - frames[-self.min_frames_consecutivos]
                            == self.min_frames_consecutivos - 1
                            and not self.node_history[best_node_id]["in_graph"]
                        ):
                            # Obtener la estancia en la que se encuentra
                            if name == "refrigerator" and habitacion != "cocina":
                                self.node_history.pop(best_node_id, None)
                                continue

                            # Crear nodo
                            inst_id = self.node_history[best_node_id][
                                "class_instance_id"
                            ]
                            node_name = f"{name}_{inst_id}"
                            pose_str = f"[{X}, {Y}, {Z}]"

                            # Crear nodo y arista
                            self.graph.create_node(name=node_name, type_="Object")
                            if habitacion != "desconocido":
                                self.graph.create_edge(
                                    type_="at",
                                    source_node=node_name,
                                    target_node=habitacion,
                                )

                            node = self.graph.get_node(node_name)
                            node.set_property(key="position", value=pose_str)

                            self.node_history[best_node_id]["in_graph"] = True

                    if self.node_history[best_node_id]["in_graph"]:
                        nodos_activos[best_node_id] = (box, [X, Y, Z], name, habitacion)

                # Es un objeto nuevo, actualizar informacion
                else:
                    print("Nuevo objeto observado")
                    nuevo_id = self.next_node_id
                    self.next_node_id += 1

                    self.node_history[nuevo_id]["cls_id"] = id_clase
                    self.node_history[nuevo_id]["frames"].append(frame_count)
                    self.node_history[nuevo_id]["position_3d"] = [X, Y, Z]
                    self.node_history[nuevo_id]["track_id"] = track_id
                    self.node_history[nuevo_id]["score"] = score

                    if embeddings_propuestos is not None:
                        self.node_history[nuevo_id]["embeddings"].append(
                            embeddings_propuestos
                        )

                    class_instance_id = self.next_instance_per_class[int(id_clase)]
                    self.next_instance_per_class[int(id_clase)] += 1
                    self.node_history[nuevo_id]["class_instance_id"] = class_instance_id

                    self.node_history[nuevo_id]["diagonal"] = t_objeto

                    self.node_history[nuevo_id]["raw_position_3d"] = [X, Y, Z]

        return nodos_activos

    def relacion_vertical(self, box1, box2, pose1, pose2):
        """
        Determina si obj1 esta encima o debajo de obj2.
        """
        x1_s, y1_s, x2_s, y2_s = box1
        x1_t, y1_t, x2_t, y2_t = box2
        X1, Y1, Z1 = pose1
        X2, Y2, Z2 = pose2

        # Distancia en el plano horizontal
        distancia_xy = ((X1 - X2) ** 2 + (Y1 - Y2) ** 2) ** 0.5
        umbral_distancia_xy = 0.4

        if distancia_xy > umbral_distancia_xy:
            return None

        # Overlap horizontal en 2D
        overlap_x = max(0, min(x2_s, x2_t) - max(x1_s, x1_t))
        min_width = min(x2_s - x1_s, x2_t - x1_t)
        ratio_overlap = overlap_x / (min_width + 1e-6)

        if ratio_overlap < 0.15:
            return None

        # Calcular diferencias de altura
        diff_altura_3d = Z1 - Z2

        cy1_2d = (y1_s + y2_s) / 2
        cy2_2d = (y1_t + y2_t) / 2
        h_avg = ((y2_s - y1_s) + (y2_t - y1_t)) / 2
        diff_altura_2d = (cy2_2d - cy1_2d) / (h_avg + 1e-6)

        # Umbrales logicos
        umbral_altura_3d = 0.05
        umbral_altura_2d = 0.3

        voto_3d = None
        if diff_altura_3d >= umbral_altura_3d:
            voto_3d = "encima_de"
        elif diff_altura_3d <= -umbral_altura_3d:
            voto_3d = "debajo_de"

        voto_2d = None
        if diff_altura_2d >= umbral_altura_2d:
            voto_2d = "encima_de"
        elif diff_altura_2d <= -umbral_altura_2d:
            voto_2d = "debajo_de"

        # Acuerdo mutuo
        if voto_3d is not None and voto_2d is not None and voto_3d == voto_2d:
            return voto_3d

        # Si la diferencia 3D es muy pronunciada (> 20cm), confiar puramente en la métrica espacial
        if voto_3d is not None and abs(diff_altura_3d) >= 0.20:
            return voto_3d

        return None

    def extraer_relaciones(self, nodos, frame):
        """
        Funcion para extraer las relaciones espaciales de los objetos presentes en el frame actual
        """

        lista_ids = list(nodos.keys())

        # Para todos los objetos
        for i in range(len(lista_ids)):
            for j in range(i + 1, len(lista_ids)):
                rel = None

                id1 = lista_ids[i]
                id2 = lista_ids[j]

                box1, pose1, name_s, _ = nodos[id1]
                box2, pose2, name_t, _ = nodos[id2]

                x1_s, y1_s, x2_s, y2_s = map(float, box1.cpu().numpy())
                x1_t, y1_t, x2_t, y2_t = map(float, box2.cpu().numpy())

                rel = self.relacion_vertical(
                    (x1_s, y1_s, x2_s, y2_s), (x1_t, y1_t, x2_t, y2_t), pose1, pose2
                )

                if rel is not None:
                    source_inst_id = self.node_history[id1]["class_instance_id"]
                    source = f"{name_s}_{source_inst_id}"

                    target_inst_id = self.node_history[id2]["class_instance_id"]
                    target = f"{name_t}_{target_inst_id}"

                    # Comprobar si ya existe una relacion que contradiga a la que se va a generar
                    aristas_ida = self.graph.get_edges_between_nodes(source, target)
                    aristas_vuelta = self.graph.get_edges_between_nodes(target, source)

                    relaciones_contrarias = {
                        "encima_de": "debajo_de",
                        "debajo_de": "encima_de",
                    }

                    contraria = relaciones_contrarias.get(rel)

                    if (
                        rel not in aristas_ida
                        and contraria not in aristas_ida
                        and rel not in aristas_vuelta
                        and contraria not in aristas_vuelta
                    ):
                        self.graph.create_edge(
                            type_=rel, source_node=source, target_node=target
                        )

        # Limpiar memoria
        self.eliminar_nodos()

        self.mostrar_frame(nodos, frame)

    def mostrar_frame(self, nodos, frame):
        """
        Funcion para visualizar el frame actual junto con las detecciones realizadas sobre este
        """

        for node_id, datos in nodos.items():
            box, _, name, __ = datos
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            cls_id = self.node_history[node_id]["cls_id"]
            inst_id = self.node_history[node_id]["class_instance_id"]
            label = f"{name}_{inst_id}"

            color = self.colores_clase[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Vision robot", frame)
        cv2.waitKey(1)

    def eliminar_nodos(self):
        """
        Funcion para eliminar los objetos que no se han convertido en nodos y fueron observados hace demasiado
        """

        eliminar = []

        for id, datos in self.node_history.items():
            frames = datos["frames"]

            # Eliminar propuesta si lleva mas de 30 frames sin aparecer y no esta en grafo
            if (
                len(frames) > 0
                and self.frame_count - frames[-1] >= 30
                and not datos["in_graph"]
            ):
                eliminar.append(id)

        for id in eliminar:
            self.node_history.pop(id, None)


def main():
    if not rclpy.ok():
        rclpy.init()

    node = CrearGrafo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
