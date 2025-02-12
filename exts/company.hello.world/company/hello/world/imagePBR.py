import numpy as np
import cv2
from pxr import Sdf, Usd, UsdGeom, UsdShade
import omni.kit.commands
 
def apply_texture_to_prim(prim_path: str, image):
    """
    将纹理（从 get_image_func 获取）应用到 Omniverse 中的一个 prim。


    Args:
    prim_path: 要应用纹理的 prim 的路径（例如，"/World/Cube"）。
    get_image_func: 返回表示图像的 NumPy 数组的函数。
    """
    # 1. 获取 Stage 和 Prim
    stage = Usd.Stage.GetCurrent()
    prim = stage.GetPrimAtPath(prim_path)

    # 3. 创建材质和着色器 (UsdPreviewSurface)
    material_path = f"{prim_path}/Looks/GeneratedMaterial"  # 材质路径
    shader_path = f"{material_path}/Shader"  # 着色器路径

    
    from pxr import Vt
        
    # 检查图像数据类型以确定适当的 VtArray 类型
    if image.dtype == np.uint8:  # 假设为 8 位颜色通道
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB 图像
            vt_array = Vt.Vec3fArray([(float(r) / 255, float(g) / 255, float(b) / 255) for r, g, b in image.reshape(-1, 3)])
        elif len(image.shape) == 2:  # 灰度图像
            vt_array = Vt.FloatArray([float(val) / 255 for val in image.flatten()])
        else:
            raise ValueError("不支持的图像格式")
    else:
        raise ValueError("不支持的图像数据类型")
 

    # 检查材质是否已存在
    material = UsdShade.Material.Get(stage, material_path)
    if not material:
        material = UsdShade.Material.Define(stage, material_path)


    shader = UsdShade.Shader.Get(stage, shader_path)
    if not shader:
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        # 设置着色器输入（例如，漫反射颜色） - 这部分取决于你如何使用图像
        # 为简单起见，我们假设它是一个漫反射纹理。你可能需要根据你的材质设置进行调整。
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))  # 默认白色漫反射
        
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")


    # 4. 创建 UV 纹理
    uv_texture_path = f"{material_path}/UVTexture"
    uv_texture = UsdShade.Shader.Get(stage, uv_texture_path)
    if not uv_texture:
        uv_texture = UsdShade.Shader.Define(stage, uv_texture_path)
        uv_texture.CreateIdAttr("UsdUVTexture")


    # 5. 将纹理数据加载到 UsdUVTexture 中
    # 转换 numpy 数组为 VtArray
    # 类型应与图像数据匹配。如果是 RGB，则应为 Vt.Vec3fArray


    # 创建纹理资源的路径
    texture_asset_path = f"{material_path}/UVTexture_asset"  # 纹理资源路径


    # 创建属性并设置值
    file_attr = uv_texture.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_attr.Set(texture_asset_path)


    # 6. 将 UV 纹理连接到着色器
    diffuse_input = shader.GetInput("diffuseColor")
    uv_texture_output = uv_texture.GetOutput("result")
    if diffuse_input and uv_texture_output:
        iffuse_input.ConnectToSource(uv_texture_output)
    else:
        print("无法将 UV 纹理连接到着色器。")
        return
    

    # 7. 应用材质绑定 API
    geom = UsdGeom.GetPrim(stage, prim_path)
    if geom:
        binding_api = UsdShade.MaterialBindingAPI.Apply(geom)
        binding_api.Bind(material)
    else:
        print(f"无法将材质绑定到 {prim_path} 的 prim")
        return
 

    print(f"成功将纹理应用到 {prim_path} 的 prim")
 
 

#  # 示例用法：
#  def get_image():
#   img_path = "path/to/your/image.png"  # 替换为你的图像路径
#   try:
#   image = np.array(cv2.imread(img_path))
#   return image
#   except Exception as e:
#   print(f"读取图像时出错：{e}")
#   return None
 

 # 将 "/World/Cube" 替换为你的 prim 的实际路径
#  apply_texture_to_prim("/World/Cube", get_image)
