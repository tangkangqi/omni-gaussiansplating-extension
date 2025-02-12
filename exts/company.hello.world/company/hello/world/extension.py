import omni.ext
import omni.ui as ui
from omni.ui import scene as sc
from omni.kit.viewport.utility import get_active_viewport_window, get_active_viewport, get_active_viewport_and_window
from PIL import Image
from pxr import Gf, Usd, UsdGeom
import omni.ui.scene as scene
import numpy as np
import cv2

class ImageDisplayExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        print("Image Display Extension started")

        self._window = ui.Window("Image Display", width=300, height=400)
        self.ext_id = ext_id
        with self._window.frame:
            with ui.VStack():
                self.image_path_field = ui.StringField(name="Image Path")
                self.load_button = ui.Button("Load Image", clicked_fn=self.load_and_display_image)
                self.scene_view = None

    # def _on_rendering_event(self, event):
    def load_and_display_image0(self):
        """Called by rendering_event_stream."""
        image = np.array(image) # received with shape (H*, W*, 3)
        image = cv2.resize(image, (self.rgba_w, self.rgba_h), interpolation=cv2.INTER_LINEAR) # resize to (H, W, 3)
        self.rgba[:,:,:3] = image * 255
        # self.rgba[:,:,:3] = (self.rgba[:,:,:3] + np.ones((self.rgba_h, self.rgba_w, 3), dtype=np.uint8)) % 256
        self.ui_nerf_provider.set_bytes_data(self.rgba.flatten().tolist(), (self.rgba_w, self.rgba_h))

    def load_and_display_image(self):
        image_path = self.image_path_field.model.as_string
        image_path = "/home/dgxsa/Desktop/tkq/threegpt/gsplat-kit/kit-exts-project/exts/company.hello.world/company/hello/world/icon.png"
        image_path = "/home/dgxsa/Desktop/frame_00001.png"

        try:
            img = Image.open(image_path)
            img = img.convert("RGBA")
            width, height = img.size
            # img_data = img.tobytes("raw", "RGBA")
            img_data = img

            class MyImageProvider(ui.ImageProvider):
                def __init__(self, width, height, data):
                    super().__init__()

                    self._width = width
                    self._height = height
                    self._data = data
                    

                def request_texture(self):
                    return self._width, self._height, self._data

            viewport_window = get_active_viewport_window()
            resolution = viewport_window.viewport_api.resolution
            width, height = resolution

            img = cv2.imread(image_path)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            rgba = np.ones((height, width, 4), dtype=np.uint8) * 128
            """RGBA image buffer. The shape is (H, W, 4), following the NumPy convention."""
            rgba[:,:,3] = 255
            rgba[:,:,:3] = img * 255

            ui_nerf_provider = ui.ByteImageProvider()
            # ui_nerf_img = ui.ImageWithProvider(
            #         ui_nerf_provider,
            #         width=ui.Percent(100),
            #         height=ui.Percent(100),
            #     )
            ui_nerf_provider.set_bytes_data(rgba.flatten().tolist(), (width, height))


            image_provider = MyImageProvider(width, height, img_data)

            viewport_api = get_active_viewport()
            camera_to_world_mat: Gf.Matrix4d = viewport_api.transform
            object_to_world_mat: Gf.Matrix4d = Gf.Matrix4d()
        
            dire_prim_path = "/World/Cube"
            stage: Usd.Stage = omni.usd.get_context().get_stage()
            selected_prim: Usd.Prim = stage.GetPrimAtPath(dire_prim_path)
            selected_xform: UsdGeom.Xformable = UsdGeom.Xformable(selected_prim)
            # object_to_world_mat = selected_xform.GetLocalTransformation()
            # Get the active viewport window
                    

            # with window.get_frame("image_frame"):
            #     self.scene_view = sc.SceneView()
            #     with self.scene_view.scene:
            #         sc.Image(image_path, width=width, height=height)
            #     viewport.add_scene_view(scene_view)
            
            if viewport_window:
                # Create a unique frame for our SceneView
                with viewport_window.get_frame(self.ext_id):
                    # Create a default SceneView
                    self.scene_view = sc.SceneView()
                    with self.scene_view.scene:
                        # Display image in viewport
                        # sc.Image(image_provider)
                        sc.Image(ui_nerf_provider, width=width, height=height)
                        # sc.Image(image_provider=image_path, width=width, height=height)
                        # sc.Image(image_provider=image_path, width=width, height=height)
                        pass
            # sc.Image(image_provider=image_path, width=width, height=height)
            # sc.Image(image_provider=image_path)

        except Exception as e:
            print(f"Error loading image: {e}")

    def on_shutdown(self):
        print("Image Display Extension shutting down")
        if self.scene_view:
            self.scene_view.destroy()
            self.scene_view = None
        if self._window:
            self._window.destroy()
            self._window = None