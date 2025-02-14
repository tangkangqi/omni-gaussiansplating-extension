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

    async def periodic_update(self):
        while True:
            await asyncio.sleep(0.05)  # 每隔 0.1 秒检查一次
            if self.interactive_flag !=1:
                return
            self.get_transform()
            if self.previous_rotation != self.camera_rotation:
                # if self.is_camera_changed():
                self.previous_rotation = self.camera_rotation
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
                with ui.HStack():
                    self.load_button = ui.Button("Load Image", clicked_fn=self.load_and_display_image)
                    self.clear_button = ui.Button("Clear Image", clicked_fn=self.clear_scene_image)
                    self.tgds_button = ui.Button("Start Intercative tdgs", clicked_fn = self.set_interactive)
                    if self.interactive_flag == 1:
                        self._task = asyncio.ensure_future(self.periodic_update())
        # self.rendering_event_stream = self.usd_context.get_rendering_event_stream()
        # self.rendering_event_delegate = self.rendering_event_stream.create_subscription_to_pop(
        #     self.load_and_display_image
        # )

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
        
        # self.camera_label.text = f"""camera_to_world_mat: {camera_to_world_mat}, \n 
        # object_to_world_mat: {object_to_world_mat}, \n 
        # camera_to_object_pos: {camera_to_object_pos}, \n
        # camera_to_object_rot: {camera_to_object_rot}"""

        self.camera_label.text = f"""object_to_world_mat: {object_to_world_mat}, \n 
        camera_to_object_pos: {camera_to_object_pos}, \n
        camera_to_object_rot: {camera_to_object_rot}"""

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
        
        resolution = self.viewport_window.viewport_api.resolution
        width, height = resolution

        img = cv2.imread(image_path)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        rgba = np.ones((height, width, 4), dtype=np.uint8) * 128
        """RGBA image buffer. The shape is (H, W, 4), following the NumPy convention."""
        rgba[:,:,3] = 255
        rgba[:,:,:3] = img * 255        
        self.tgs_provider.set_bytes_data(rgba.flatten().tolist(), (width, height))
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
                    sc.Image(self.tgs_provider, width=2, height=2)
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