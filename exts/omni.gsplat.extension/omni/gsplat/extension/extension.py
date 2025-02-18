import omni.ext
import omni.ui as ui
from omni.ui import scene as sc
from omni.kit.viewport.utility import get_active_viewport_window, get_active_viewport, get_active_viewport_and_window
from PIL import Image
from pxr import Gf, Usd, UsdGeom
# import omni.ui.scene as scene
import numpy as np
import cv2
import asyncio
import torch
import sys
import requests
import os

# import plyfile
print(sys.executable)
print(sys.version)
print(torch.__file__)
print(torch.__version__)


class ImageDisplayExtension(omni.ext.IExt):
    def __init__(self):
        super().__init__()
        """The Python version must match the backend version for RPyC to work."""
        self.camera_position: Gf.Vec3d = None
        self.camera_rotation: Gf.Vec3d = None
        self.viewport_window = get_active_viewport_window()
        self.tgs_provider = ui.ByteImageProvider()
        self.viewport_api = get_active_viewport()
        self.usd_context = omni.usd.get_context()
        self.scene_view = None
        self.previous_rotation = None
        self.interactive_flag = 0
        self.camera_fov = 0.6
        self.camera_info = {'pos': self.camera_position, 'rot': self.camera_rotation, 'width': 0, 'height': 0, 'fov': self.camera_fov}
        self.cv2_image = None
        

    async def periodic_update(self):
        while True:
            await asyncio.sleep(0.01)  # 每隔 0.1 秒检查一次
            if self.interactive_flag !=1:
                return
            self.get_transform()
            if self.previous_rotation != self.camera_rotation or self.previous_position != self.camera_position or self.camera_fov != self.previous_fov:
                # if self.is_camera_changed():
                self.previous_rotation = self.camera_rotation
                self.previous_position = self.camera_position
                self.previous_fov = self.camera_fov
                self.load_and_display_image()

                
    def set_interactive(self):
        self.interactive_flag = 1
        asyncio.ensure_future(self.periodic_update())

    def on_startup(self, ext_id):
        print("Image Display Extension started")
        self.check_and_create_cube()
        self._window = ui.Window("Image Display", width=300, height=400)
        self.ext_id = ext_id
        with self._window.frame:
            with ui.VStack():
                self.image_path_field = ui.StringField(name="Image Path")
                self.camera_label = ui.Label('camera status')
                # self.slider = ui.Slider(min_value=0.0, max_value=100.0, value=0.0)
                # 监听滑块的值变化
                # self.slider.set_on_value_changed_fn(self.on_slider_changed)
                with ui.HStack(height=20):
                    ui.Label("set camera fov:", width=50)
                    ui.Spacer(width=10)
                    self.float_model = ui.SimpleFloatModel(0.01, min=0.01,max=4)
                    example_progress = ui.ProgressBar(model=self.float_model, width=200, height=10)
                    new_style = {"color" : 0xFF00b976}
                    example_progress.set_style(new_style)
                    ui.Spacer(width=10)
                    ui.FloatSlider(self.float_model, width=200 ,min=0.01,max=4,step=0.01)

                with ui.HStack():
                    self.load_button = ui.Button("Load Image", clicked_fn=self.load_and_display_image)
                    self.clear_button = ui.Button("Clear Image", clicked_fn=self.clear_scene_image)
                    self.tgds_button = ui.Button("Start Intercative tdgs", clicked_fn = self.set_interactive)
                    self.focal_button = ui.Button("Initial Focal", clicked_fn = self.set_cube_focal)
                    self.screenshot_button = ui.Button("Save Image", clicked_fn = self.save_gs_image)
                    if self.interactive_flag == 1:
                        self._task = asyncio.ensure_future(self.periodic_update())
        # self.rendering_event_stream = self.usd_context.get_rendering_event_stream()
        # self.rendering_event_delegate = self.rendering_event_stream.create_subscription_to_pop(
        #     self.load_and_display_image
        # )

    def save_gs_image(self):
        fd_path = os.path.dirname(os.path.abspath(__file__))
        cv2.imwrite(os.path.join(fd_path,'test_feteched_gs.png'), self.cv2_image, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

    def on_slider_changed(self, value):
        """当滑块值变化时更新进度条和标签"""
        self.progress_bar.value = value
        # self.label.text = f"Current Value: {value:.2f}"
        self.camera_fov = float(f"{value:.2f}")

    def check_and_create_cube(self):
        stage = self.usd_context.get_stage()
        if not stage:
            print("[INFO] No USD Stage found")
            return

        cube_path = "/World/Cube"
        cube_prim = stage.GetPrimAtPath(cube_path)

        if not cube_prim.IsValid():
            print("[INFO] Cube not found, creating...")
            omni.kit.commands.execute('CreateMeshPrimWithDefaultXform', prim_type='Cube', prim_name='Cube', select_new_prim=False, prim_path=cube_path)
        else:
            print("[INFO] Cube already exists")

    def update_ui(self):
        print("[omni.nerf.viewport] Updating UI")
        # Ref: https://forums.developer.nvidia.com/t/refresh-window-ui/221200
        self.load_and_display_image()

    def get_transform(self):
        camera_to_world_mat: Gf.Matrix4d = self.viewport_api.transform
        object_to_world_mat: Gf.Matrix4d = Gf.Matrix4d()
        
        dire_prim_path = "/World/Cube"
        stage: Usd.Stage = self.usd_context.get_stage()
        selected_prim: Usd.Prim = stage.GetPrimAtPath(dire_prim_path)

        selected_xform: UsdGeom.Xformable = UsdGeom.Xformable(selected_prim)
        object_to_world_mat = selected_xform.GetLocalTransformation()
        # In USD, pre-multiplication is used for matrices.
        # Ref: https://openusd.org/dev/api/usd_geom_page_front.html#UsdGeom_LinAlgBasics
        world_to_object_mat: Gf.Matrix4d = object_to_world_mat.GetInverse()
        camera_to_object_mat: Gf.Matrix4d = camera_to_world_mat * world_to_object_mat
        camera_to_object_pos: Gf.Vec3d = camera_to_object_mat.ExtractTranslation()
        camera_to_object_mat.Orthonormalize()
        camera_to_object_rot: Gf.Vec3d = Gf.Vec3d(*reversed(camera_to_object_mat.ExtractRotation().Decompose(*reversed(Gf.Matrix3d()))))
        if camera_to_object_pos != self.camera_position or camera_to_object_rot != self.camera_rotation:
            self.camera_position = camera_to_object_pos
            self.camera_rotation = camera_to_object_rot
        
        self._viewport = get_active_viewport()
        camera = self._viewport.get_active_camera()
        # 获取视野角度 (fov)
        # fov = self._viewport.get_active_camera_fov()
        # fov = np.sum(cam)
        # fov = 0.69111120
        self.camera_fov = self.float_model.get_value_as_float()
        fov = self.camera_fov
        # 获取视口的宽高比
        # aspect_ratio = self._viewport.get_viewport_rect().size[0] / self._viewport.get_viewport_rect().size[1]
        aspect_ratio = None

        # scales = selected_xform.GetLocalTransformation().ExtractScale()

        xform = UsdGeom.Xform(selected_prim)
        localToWorld = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform(localToWorld)
        rotation = transform.GetRotation()
        scale = Gf.Vec3f(transform.GetScale())


        # self.camera_label.text = f"""camera_to_world_mat: {camera_to_world_mat}, \n 
        # object_to_world_mat: {object_to_world_mat}, \n 
        # camera_to_object_pos: {camera_to_object_pos}, \n
        # camera_to_object_rot: {camera_to_object_rot}"""

        resolution = self.viewport_window.viewport_api.resolution
        width, height = resolution

        self.camera_label.text = f"""object_to_world_mat: {object_to_world_mat}, \n 
        camera_to_object_pos: {camera_to_object_pos}, \n
        camera_to_object_rot: {camera_to_object_rot}, \n
        fov, aspect_ratio: {fov}, {aspect_ratio}, \n
        resolution: {resolution},\n
        scale: {scale}"""

        self.camera_info = {
            "rot": [v for v in camera_to_object_rot],
            "pos": [v for v in camera_to_object_pos],
            "width": width,
            "height": height,
            "fov":fov
        }
        return self.camera_info

    def set_cube_focal(self):
        rot = [1,0,0]
        pos = [1,1,1]
        stage = self.usd_context.get_stage()
        cube_path = "/World/Cube"
        cube_prim = stage.GetPrimAtPath(cube_path)

        xform = UsdGeom.Xformable(cube_prim)
        xform.MakeMatrixXform()

        transform_api = UsdGeom.XformCommonAPI(cube_prim)
        time = Usd.TimeCode.Default() 
        world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
        translation: Gf.Vec3d = world_transform.ExtractTranslation()

        # 转换旋转角度为弧度并创建旋转矩阵
        rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rot[0]) *
                                                  Gf.Rotation(Gf.Vec3d(0, 1, 0), rot[1]) *
                                                  Gf.Rotation(Gf.Vec3d(0, 0, 1), rot[2]))
        # rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rot[0]) *
        #                                           Gf.Rotation(Gf.Vec3d(0, 1, 0), rot[1]) *
        #                                           Gf.Rotation(Gf.Vec3d(0, 0, 1), rot[2]))
        
        # 设置 Transform 矩阵（位置 + 旋转）
        transform_matrix = rotation_matrix.SetTranslateOnly(Gf.Vec3d(*pos))
        # xform.SetLocalTransformation(transform_matrix)
        # transform_api.SetTranslate(transform_matrix)
        transform_api.SetTranslate(rot)

    def get_prim_camera(self):
        stage = self.usd_context.get_stage()
        prim = stage.GetPrimAtPath("/World/Cube")
        if not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        # transform = xformable.ComputeLocalToWorldTransform()
        
        # Position
        position = Gf.Vec3d(transform.ExtractTranslation())
        
        # Rotation (Euler angles)
        rotation = transform.ExtractRotation()
        # euler_angles = rotation.GetEulerAngles()
        euler_angles = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    
        # Scale
        # scale = Gf.Vec3d(xformable.GetLocalTransformation().ExtractScale())
        scale = Gf.Vec3d(*(v.GetLength() for v in xformable.GetLocalTransformation().ExtractRotationMatrix()))
        self.camera_label.text = f"Scale: {scale}, Rotation: {euler_angles}, Position: {position}"

    def get_world_transform_xform(self):
        """
        Get the local transformation of a prim using Xformable.
        See https://openusd.org/release/api/class_usd_geom_xformable.html
        Args:
            prim: The prim to calculate the world transformation.
        Returns:
            A tuple of:
            - Translation vector.
            - Rotation quaternion, i.e. 3d vector plus angle.
            - Scale vector.
        """

        self._viewport = get_active_viewport()
        camera = self._viewport.get_active_camera()

        transform = UsdGeom.Xformable(camera).ComputeWorldToLocalTransform(Usd.TimeCode.DEFAULT)
        position = transform.ExtractTranslation()
        rotation = transform.ExtractRotation()
        euler_angles = rotation.GetEulerAngles()


        # 获取视野角度 (fov)
        fov = self._viewport.get_active_camera_fov()


        # 获取视口的宽高比
        aspect_ratio = self._viewport.get_viewport_rect().size[0] / self._viewport.get_viewport_rect().size[1]

        self.camera_label.text = f"Position: {position}, Rotation: {euler_angles}, Field of View: {fov}, Aspect Ratio: {aspect_ratio}"


    def get_cube_info(self):
        # if not prim.IsValid():
        #     return
        # xform = UsdGeom.Xformable(prim)
        # time = Usd.TimeCode.Default() # The time at which we compute the bounding box
        # world_transform: Gf.Matrix4d = xform.ComputeLocalToWorldTransform(time)
        # translation: Gf.Vec3d = world_transform.ExtractTranslation()
        # rotation: Gf.Rotation = world_transform.ExtractRotation()
        # scale: Gf.Vec3d = Gf.Vec3d(*(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
        stage = self.usd_context.get_stage()
        prim = stage.GetPrimAtPath("/World/Cube")

        velocity = Gf.Vec3f(500.0,  0.0, 0.0)
        xform = UsdGeom.Xform(prim)
        localToWorld = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        transform = Gf.Transform(localToWorld)
        rotation = transform.GetRotation()
        scale = Gf.Vec3f(transform.GetScale())
        velocity = rotation.TransformDir(velocity)
        velocity = Gf.CompMult(scale, velocity)

        # translation: Gf.Vec3d = transform.ExtractTranslation()
        translation = 0

        self.camera_label.text = f"Scale: {scale}, Rotation: {rotation}, Position: {translation}"
        return translation, rotation, scale

    def load_and_display_image(self):
    # def load_and_display_image(self, event):
        self.get_transform()
        image_path = self.image_path_field.model.as_string
        # image_path = "/home/dgxsa/Desktop/tkq/threegpt/gsplat-kit/kit-exts-project/exts/company.hello.world/company/hello/world/icon.png"
        # image_path = "/home/dgxsa/Desktop/frame_00001.png"
        img_id = int(self.camera_rotation[0] )%10
        image_path = f"/home/dgxsa/Desktop/tkq/threegpt/data/nerfstudio/poster/images/frame_0002{img_id}.png"
        # image_path = f"/mnt/threegpt/data/nerfstudio/poster/images/frame_0002{img_id}.png"

        data = self.camera_info
        width, height = data['width'], data['height']
        url = "http://127.0.0.1:8892/get_render"
        response = requests.post(url, json=data)
        jpeg_data = np.frombuffer(response.content, dtype=np.uint8)
        decoded_image = cv2.imdecode(jpeg_data, cv2.IMREAD_UNCHANGED)

        # img = cv2.imread(image_path)
        img = decoded_image
        self.cv2_image = img
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        rgba = np.ones((height, width, 4), dtype=np.uint8) * 128
        """RGBA image buffer. The shape is (H, W, 4), following the NumPy convention."""
        rgba[:,:,3] = 255
        rgba[:,:,:3] = img * 255       
        rgba_list =  rgba.flatten().tolist()

        self.tgs_provider.set_bytes_data(rgba_list, (width, height))
        # self.camera_label.text = "%s, %s,%s"%(str(self.camera_position), str(height),str(width))
        # self.get_prim_camera()
        # self.get_world_transform_xform()

        if self.viewport_window:
            # Create a unique frame for our SceneView
            with self.viewport_window.get_frame(self.ext_id):
                # Create a default SceneView                
                # if self.scene_view:
                #     self.scene_view.destroy()
                # self.scene_view = sc.SceneView()
                # self.scene_view = sc.SceneView(aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT)
                self.scene_view = sc.SceneView(aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH)
                self.scene_view = sc.SceneView(screen_aspect_ratio=0)
                with self.scene_view.scene:
                    self.scene_view.scene.clear()
                    # sc.Image(self.tgs_provider, width=width, height=height)
                    sc.Image(self.tgs_provider, width=2, height=2*height/width)
                    # sc.Image(self.tgs_provider, width=2*width/height, height=2)
                    # sc.Image(self.tgs_provider)
                # pass
    def clear_scene_image(self):
        self.interactive_flag = -1 
        with self.viewport_window.get_frame(self.ext_id):
            self.scene_view = sc.SceneView(aspect_ratio_policy=sc.AspectRatioPolicy.STRETCH)
            self.scene_view = sc.SceneView(screen_aspect_ratio=0)
            with self.scene_view.scene:
                self.scene_view.scene.clear() # 清空 scene，恢复正常视口

    def on_shutdown(self):
        print("Image Display Extension shutting down")
        if self.scene_view:
            self.scene_view.destroy()
            self.scene_view = None
        if self._window:
            self._window.destroy()
            self._window = None