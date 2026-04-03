import importlib
import socket
import time
import base64
from functools import lru_cache
from pathlib import Path
from vuer import Vuer
from vuer.events import ClientEvent
from vuer.schemas import (
    Box,
    Cylinder,
    DefaultScene,
    Group,
    Hands,
    ImageBackground,
    Movable,
    SkeletalGripper,
    Sphere,
    Stl,
    WebRTCStereoVideoPlane,
)
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
import numpy as np
import asyncio

MODULE_DIR = Path(__file__).resolve().parent
XR_HAND_GRIPPER_OFFSET = [0.0, -0.075, 0.0]
XR_HAND_GRIPPER_ROTATION = [0.0, 0.0, float(np.pi)]
XR_LEFT_GRIPPER_COLOR = "#8ED6FF"
XR_RIGHT_GRIPPER_COLOR = "#FFB3B3"
XR_LEFT_HAND_COLOR = "#DDE7F4"
XR_RIGHT_HAND_COLOR = "#F2D6D6"
XR_DEFAULT_PINCH_WIDTH = 0.03
XR_MIN_PINCH_WIDTH = 0.008
XR_MAX_PINCH_WIDTH = 0.045
XR_THUMB_TIP_INDEX = 4
XR_INDEX_TIP_INDEX = 9


@lru_cache(maxsize=None)
def _file_to_data_url(path_str: str, mime_type: str) -> str:
    data = Path(path_str).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

def _load_webrtc_symbols():
    """Load WebRTC server symbols only when WebRTC mode is requested."""
    module = importlib.import_module("webrtc.zed_server")
    names = [
        "Args",
        "RTC",
        "on_shutdown",
        "index",
        "javascript",
        "logging",
        "ssl",
        "web",
        "aiohttp_cors",
    ]
    return {name: getattr(module, name) for name in names}


def _find_free_port(start_port, max_tries=32):
    """Find an available TCP port starting from start_port."""
    port = int(start_port)
    for _ in range(max_tries):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            return port
        except OSError:
            port += 1
        finally:
            sock.close()
    raise RuntimeError(f"Cannot find a free port near {start_port}")


def _resolve_local_file(path_str):
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.as_posix()
    candidate = MODULE_DIR / path
    if candidate.exists():
        return candidate.as_posix()
    return path.as_posix()

class OpenTeleVision:
    def __init__(
        self,
        img_shape,
        shm_name,
        queue,
        toggle_streaming,
        stream_mode="image",
        cert_file="./cert.pem",
        key_file="./key.pem",
        ngrok=False,
        vuer_port=8012,
    ):
        # self.app=Vuer()
        self.img_shape = (img_shape[0], 2*img_shape[1], 3)
        self.img_height, self.img_width = img_shape[:2]
        selected_port = _find_free_port(vuer_port)
        if selected_port != int(vuer_port):
            print(f"[*] Vuer port {vuer_port} busy, switched to {selected_port}")

        if ngrok:
            self.app = Vuer(host='0.0.0.0', port=selected_port, queries=dict(grid=False), queue_len=3)
        else:
            cert_file = _resolve_local_file(cert_file)
            key_file = _resolve_local_file(key_file)
            self.app = Vuer(
                host='0.0.0.0',
                port=selected_port,
                cert=cert_file,
                key=key_file,
                queries=dict(grid=False),
                queue_len=3,
            )

        self.app.add_handler("HAND_MOVE")(self.on_hand_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        if stream_mode == "image":
            existing_shm = shared_memory.SharedMemory(name=shm_name)
            self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
            self.app.spawn(start=False)(self.main_image)
        elif stream_mode == "webrtc":
            self.app.spawn(start=False)(self.main_webrtc)
        else:
            raise ValueError("stream_mode must be either 'webrtc' or 'image'")

        self.left_hand_shared = Array('d', 16, lock=True)
        self.right_hand_shared = Array('d', 16, lock=True)
        self.left_landmarks_shared = Array('d', 75, lock=True)
        self.right_landmarks_shared = Array('d', 75, lock=True)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)
        self._xr_assets = self._load_xr_assets()
        if stream_mode == "webrtc":
            try:
                webrtc = _load_webrtc_symbols()
            except Exception as e:
                raise ImportError(
                    f"WebRTC mode requires optional dependency webrtc.zed_server: {e}"
                ) from e

            # webrtc server
            if webrtc["Args"].verbose:
                webrtc["logging"].basicConfig(level=webrtc["logging"].DEBUG)
            else:
                webrtc["logging"].basicConfig(level=webrtc["logging"].INFO)
            webrtc["Args"].img_shape = img_shape
            # Args.shm_name = shm_name
            webrtc["Args"].fps = 60

            ssl_context = webrtc["ssl"].SSLContext()
            ssl_context.load_cert_chain(cert_file, key_file)

            app = webrtc["web"].Application()
            cors = webrtc["aiohttp_cors"].setup(app, defaults={
                "*": webrtc["aiohttp_cors"].ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            })
            rtc = webrtc["RTC"](img_shape, queue, toggle_streaming, 60)
            app.on_shutdown.append(webrtc["on_shutdown"])
            cors.add(app.router.add_get("/", webrtc["index"]))
            cors.add(app.router.add_get("/client.js", webrtc["javascript"]))
            cors.add(app.router.add_post("/offer", rtc.offer))

            self.webrtc_process = Process(
                target=webrtc["web"].run_app,
                args=(app,),
                kwargs={"host": "0.0.0.0", "port": 8080, "ssl_context": ssl_context},
            )
            self.webrtc_process.daemon = True
            self.webrtc_process.start()
            # web.run_app(app, host="0.0.0.0", port=8080, ssl_context=ssl_context)

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    
    def run(self):
        # Avoid hard failure when default port 8012 is already occupied.
        self.app.run(free_port=True)

    async def on_cam_move(self, event, session, fps=60):
        # only intercept the ego camera.
        # if event.key != "ego":
        #     return
        try:
            # with self.head_matrix_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            # with self.aspect_shared.get_lock():
            #     self.aspect_shared.value = event.value['camera']['aspect']
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass
        # self.head_matrix = np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F")
        # print(np.array(event.value["camera"]["matrix"]).reshape(4, 4, order="F"))
        # print("camera moved", event.value["matrix"].shape, event.value["matrix"])

    async def on_hand_move(self, event, session, fps=60):
        try:
            # with self.left_hand_shared.get_lock():  # Use the lock to ensure thread-safe updates
            #     self.left_hand_shared[:] = event.value["leftHand"]
            # with self.right_hand_shared.get_lock():
            #     self.right_hand_shared[:] = event.value["rightHand"]
            # with self.left_landmarks_shared.get_lock():
            #     self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            # with self.right_landmarks_shared.get_lock():
            #     self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
            self.left_hand_shared[:] = event.value["leftHand"]
            self.right_hand_shared[:] = event.value["rightHand"]
            self.left_landmarks_shared[:] = np.array(event.value["leftLandmarks"]).flatten()
            self.right_landmarks_shared[:] = np.array(event.value["rightLandmarks"]).flatten()
        except: 
            pass
    
    async def main_webrtc(self, session, fps=60):
        session.set @ DefaultScene(
            *self._build_xr_scene_children(),
            frameloop="always",
            grid=False,
            key="default-scene",
        )
        session.upsert(
            Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False),
            to="bgChildren",
        )
        session.upsert @ WebRTCStereoVideoPlane(
                src="https://192.168.8.102:8080/offer",
                # iceServer={},
                key="zed",
                aspect=1.33334,
                height = 8,
                position=[0, -2, -0.2],
            )
        workspace_placed = False
        while True:
            xr_updates, workspace_placed = self._build_xr_dynamic_updates(workspace_placed)
            if xr_updates:
                session.upsert(xr_updates, to="children")
            await asyncio.sleep(0.03)
    
    async def main_image(self, session, fps=60):
        session.set @ DefaultScene(
            *self._build_xr_scene_children(),
            frameloop="always",
            grid=False,
            key="default-scene",
        )
        session.upsert(
            Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False),
            to="bgChildren",
        )
        workspace_placed = False
        end_time = time.time()
        while True:
            start = time.time()
            # print(end_time - start)
            # aspect = self.aspect_shared.value
            display_image = self.img_array
            xr_updates, workspace_placed = self._build_xr_dynamic_updates(workspace_placed)
            if xr_updates:
                session.upsert(xr_updates, to="children")

            # session.upsert(
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[:self.img_height],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="left-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.778,
            #     distanceToCamera=2,
            #     position=[0, -0.5, -2],
            #     rotation=[0, 0, 0],
            # ),
            # to="bgChildren",
            # )

            session.upsert(
            [ImageBackground(
                # Can scale the images down.
                display_image[::2, :self.img_width],
                # display_image[:self.img_height:2, ::2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="left-image",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height = 8,
                position=[0, -1, 3],
                # rotation=[0, 0, 0],
                layers=1,
            ),
            ImageBackground(
                # Can scale the images down.
                display_image[::2, self.img_width:],
                # display_image[self.img_height::2, ::2],
                # 'jpg' encoding is significantly faster than 'png'.
                format="jpeg",
                quality=80,
                key="right-image",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height = 8,
                position=[0, -1, 3],
                # rotation=[0, 0, 0],
                layers=2,
            )],
            to="bgChildren",
            )
            # rest_time = 1/fps - time.time() + start
            end_time = time.time()
            await asyncio.sleep(0.03)

    def _load_xr_assets(self):
        assets = {}
        try:
            left_mesh = MODULE_DIR.parent / "assets" / "inspire_hand" / "meshes" / "L_hand_base_link.STL"
            right_mesh = MODULE_DIR.parent / "assets" / "inspire_hand" / "meshes" / "R_hand_base_link.STL"
            assets["left_hand_mesh"] = _file_to_data_url(left_mesh.resolve().as_posix(), "model/stl")
            assets["right_hand_mesh"] = _file_to_data_url(right_mesh.resolve().as_posix(), "model/stl")
        except OSError as exc:
            print(f"[!] XR hand mesh preload failed: {exc}")
        return assets

    @staticmethod
    def _matrix_to_list(matrix):
        return np.asarray(matrix, dtype=np.float32).reshape(4, 4).flatten(order="F").tolist()

    @staticmethod
    def _valid_transform(matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (4, 4):
            return False
        if not np.isfinite(matrix).all():
            return False
        if abs(float(matrix[3, 3])) < 1e-6:
            return False
        return float(np.linalg.norm(matrix)) > 1e-3

    @staticmethod
    def _valid_landmarks(landmarks):
        landmarks = np.asarray(landmarks, dtype=np.float32)
        return landmarks.shape == (25, 3) and np.isfinite(landmarks).all() and float(np.linalg.norm(landmarks)) > 1e-4

    def _pinch_width(self, landmarks):
        if not self._valid_landmarks(landmarks):
            return XR_DEFAULT_PINCH_WIDTH
        thumb_tip = landmarks[XR_THUMB_TIP_INDEX]
        index_tip = landmarks[XR_INDEX_TIP_INDEX]
        pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
        return float(np.clip(0.5 * pinch_distance, XR_MIN_PINCH_WIDTH, XR_MAX_PINCH_WIDTH))

    def _workspace_root_matrix(self):
        head_matrix = self.head_matrix
        if not self._valid_transform(head_matrix):
            return None

        head_matrix = np.asarray(head_matrix, dtype=np.float32)
        head_pos = head_matrix[:3, 3]
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        forward = -head_matrix[:3, 2]
        forward[1] = 0.0
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward /= forward_norm

        right = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= right_norm

        forward = np.cross(up, right)
        forward /= max(float(np.linalg.norm(forward)), 1e-6)

        workspace_pos = head_pos + 0.55 * forward - 0.18 * up
        workspace_matrix = np.eye(4, dtype=np.float32)
        workspace_matrix[:3, 0] = right
        workspace_matrix[:3, 1] = up
        workspace_matrix[:3, 2] = forward
        workspace_matrix[:3, 3] = workspace_pos
        return self._matrix_to_list(workspace_matrix)

    def _build_hand_visual(self, side, mesh_color, gripper_color):
        children = []
        mesh_src = self._xr_assets.get(f"{side}_hand_mesh")
        if mesh_src:
            children.append(
                Stl(
                    src=mesh_src,
                    color=mesh_color,
                    opacity=0.82,
                    castShadow=True,
                    receiveShadow=True,
                    key=f"tv-{side}-hand-mesh",
                )
            )
        children.append(
            SkeletalGripper(
                key=f"tv-{side}-hand-gripper",
                color=gripper_color,
                opacity=0.95,
                position=XR_HAND_GRIPPER_OFFSET,
                rotation=XR_HAND_GRIPPER_ROTATION,
                pinchWidth=XR_DEFAULT_PINCH_WIDTH,
            )
        )
        return Group(
            *children,
            key=f"tv-{side}-hand-visual",
            hide=True,
        )

    def _build_workspace_visual(self):
        return Group(
            Box(
                args=[0.42, 0.02, 0.28],
                position=[0.0, 0.0, 0.0],
                materialType="standard",
                material=dict(color="#5C6570", roughness=0.88, metalness=0.08),
                castShadow=True,
                receiveShadow=True,
                key="tv-xr-table-top",
            ),
            Cylinder(
                args=[0.055, 0.07, 0.16, 24],
                position=[0.0, -0.09, 0.0],
                materialType="standard",
                material=dict(color="#434B55", roughness=0.92, metalness=0.1),
                castShadow=True,
                receiveShadow=True,
                key="tv-xr-table-pedestal",
            ),
            Box(
                args=[0.11, 0.01, 0.11],
                position=[0.14, 0.015, -0.01],
                materialType="standard",
                material=dict(
                    color="#8ED6FF",
                    roughness=0.45,
                    metalness=0.12,
                    transparent=True,
                    opacity=0.72,
                ),
                key="tv-xr-drop-zone",
            ),
            Movable(
                Box(
                    args=[0.065, 0.065, 0.065],
                    materialType="standard",
                    material=dict(color="#FFB347", roughness=0.32, metalness=0.06),
                    castShadow=True,
                    receiveShadow=True,
                    key="tv-xr-pickup-cube-geom",
                ),
                Sphere(
                    args=[0.012, 24, 24],
                    position=[0.0, 0.048, 0.0],
                    materialType="standard",
                    material=dict(color="#FFF7E3", emissive="#FFF7E3", emissiveIntensity=0.25),
                    key="tv-xr-pickup-cube-handle",
                ),
                key="tv-xr-pickup-cube",
                position=[-0.06, 0.05, 0.0],
                scale=0.5,
                handle=0.05,
                showFrame=False,
                localRotation=True,
            ),
            key="tv-xr-workspace",
            hide=True,
        )

    def _build_xr_scene_children(self):
        return [
            self._build_hand_visual("left", XR_LEFT_HAND_COLOR, XR_LEFT_GRIPPER_COLOR),
            self._build_hand_visual("right", XR_RIGHT_HAND_COLOR, XR_RIGHT_GRIPPER_COLOR),
            self._build_workspace_visual(),
        ]

    def _build_xr_dynamic_updates(self, workspace_placed):
        updates = []

        if not workspace_placed:
            workspace_matrix = self._workspace_root_matrix()
            if workspace_matrix is not None:
                updates.append(
                    Group(
                        key="tv-xr-workspace",
                        matrix=workspace_matrix,
                        hide=False,
                    )
                )
                workspace_placed = True

        for side, color in (("left", XR_LEFT_GRIPPER_COLOR), ("right", XR_RIGHT_GRIPPER_COLOR)):
            hand_matrix = getattr(self, f"{side}_hand")
            hand_visible = self._valid_transform(hand_matrix)
            updates.append(
                Group(
                    key=f"tv-{side}-hand-visual",
                    hide=not hand_visible,
                    **({"matrix": self._matrix_to_list(hand_matrix)} if hand_visible else {}),
                )
            )
            if hand_visible:
                updates.append(
                    SkeletalGripper(
                        key=f"tv-{side}-hand-gripper",
                        color=color,
                        opacity=0.95,
                        position=XR_HAND_GRIPPER_OFFSET,
                        rotation=XR_HAND_GRIPPER_ROTATION,
                        pinchWidth=self._pinch_width(getattr(self, f"{side}_landmarks")),
                    )
                )

        return updates, workspace_placed

    @property
    def left_hand(self):
        # with self.left_hand_shared.get_lock():
        #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def left_landmarks(self):
        # with self.left_landmarks_shared.get_lock():
        #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        # with self.right_landmarks_shared.get_lock():
            # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
            # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)

    
if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
    img_height, img_width = resolution_cropped[:2]  # 450 * 600
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    shm_name = shm.name
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
    while True:
        # print(tv.left_landmarks)
        # print(tv.left_hand)
        # tv.modify_shared_image(random=True)
        time.sleep(1)
