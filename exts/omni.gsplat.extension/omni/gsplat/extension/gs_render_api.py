import copy
import ipywidgets
import json
# import kaolin
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision
import cv2
# Gaussian splatting dependencies
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera as GSCamera

class PipelineParamsNoparse:
    """ Same as PipelineParams but without argument parser. """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # super().__init__(parser, "Pipeline Parameters")
        
def load_checkpoint(model_path, sh_degree=3, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")
    # checkpt_path = "/app/output/09ac9260-2/point_cloud/iteration_7000/point_cloud.ply"
    
    # Load guassians
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)                                                 
    return gaussians


def try_load_camera(model_path):
    """ Load one of the default cameras for the scene. """
    cam_path = os.path.join(model_path, 'cameras.json')
    if not os.path.exists(cam_path):
        print(f'Could not find saved cameras for the scene at {camp_path}; using default for ficus.')
        return GSCamera(colmap_id=0,
                        R=np.array([[-9.9037e-01,  2.3305e-02, -1.3640e-01], [ 1.3838e-01,  1.6679e-01, -9.7623e-01], [-1.6444e-09, -9.8571e-01, -1.6841e-01]]), 
                        T=np.array([6.8159e-09, 2.0721e-10, 4.03112e+00]), 
                        FoVx=0.69111120, FoVy=0.69111120, 
                        image=torch.zeros((3, 800, 800)),  # fake 
                        gt_alpha_mask=None, image_name='fake', uid=0)
        
    with open(cam_path) as f:
        data = json.load(f)
        raw_camera = data[0]
        
    tmp = np.zeros((4, 4))
    tmp[:3, :3] = raw_camera['rotation']
    tmp[:3, 3] = raw_camera['position']
    tmp[3, 3] = 1
    C2W = np.linalg.inv(tmp)
    R = C2W[:3, :3].transpose()
    T = C2W[:3, 3]
    width = raw_camera['width']
    height = raw_camera['height']
    fovx = focal2fov(raw_camera['fx'], width)
    fovy = focal2fov(raw_camera['fy'], height)
    return GSCamera(colmap_id=0,
                    R=R, T=T, FoVx=fovx, FoVy=fovy, 
                    image=torch.zeros((3, height, width)),  # fake 
                    gt_alpha_mask=None,image_name='fake', uid=0)


# model_path = 'output/6a66ab97-9'
# # model_path = '../../output/09ac9260-2/'
# gaussians = load_checkpoint(model_path)
# pipeline = PipelineParamsNoparse()
# background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
# test_camera = try_load_camera(model_path)
# render_image = (render(test_camera, gaussians, pipeline, background)['render'].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()

import math
class GaussianSplattingServer:
    def __init__(self, model_path='output/6a66ab97-9'):
        """
        初始化 GaussianSplattingServer。

        Args:
            model_path (str): 高斯模型检查点的路径。
        """
        self.model_path = model_path
        self.gaussians = self._load_gaussians()
        self.pipeline = PipelineParamsNoparse()  # 初始化渲染管线参数
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda") # 初始化背景颜色
        self.test_camera = self._load_test_camera() # 加载测试相机

    def _load_gaussians(self):
        """加载高斯模型检查点."""
        try:
            gaussians = load_checkpoint(self.model_path)
            print(f"Successfully loaded Gaussians from {self.model_path}")
            return gaussians
        except Exception as e:
            print(f"Error loading Gaussians: {e}")
            return None

    def _load_test_camera(self):
        """加载测试相机参数."""
        try:
            camera = try_load_camera(self.model_path)
            print(f"Successfully loaded test camera from {self.model_path}")
            return camera
        except Exception as e:
            print(f"Error loading test camera: {e}")
            return None

    def get_camera(self, rot=[1, 0,0], pos=[1,1,1], width=1920, height=1080, fov = 0.6):
        theta = rot
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(theta[0]), -math.sin(theta[0])],
                        [0, math.sin(theta[0]), math.cos(theta[0])]
                        ])
    
        R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                        [0, 1, 0],
                        [-math.sin(theta[1]), 0, math.cos(theta[1])]
                        ])
    
        R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                        [math.sin(theta[2]), math.cos(theta[2]), 0],
                        [0, 0, 1]
                        ])
    
        R = np.dot(R_z, np.dot(R_y, R_x))
        T = pos
        fovx = fov
        fovy = fov
        return GSCamera(colmap_id=0,
                        R=R, T=T, FoVx=fovx, FoVy=fovy, 
                        image=torch.zeros((3, height, width)),  # fake 
                        gt_alpha_mask=None,image_name='fake', uid=0)
    def get_render_image(self, current_camera = {'rot':[1, 0,0], 'pos': [1,1,1], 'width' : 1920, 'height': 1080 }):
        self.current_gs_camera = self.get_camera(current_camera['rot'], current_camera['pos'], current_camera['width'], current_camera['height'])
        try:
            render_result = render(self.current_gs_camera, self.gaussians, self.pipeline, self.background)
            render_image = (render_result['render'].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()
            return render_image
        except Exception as e:
            print(f"Error torch image: {e}")
            return None
    def get_render_cv2_image(self, current_camera = {'rot':[1, 0,0], 'pos': [1,1,1], 'width' : 1920, 'height': 1080 }):
        # self.current_gs_camera = self.get_camera(current_camera['rot'], current_camera['pos'], current_camera['width'], current_camera['height'])
        try:
            # render_result = render(self.current_gs_camera, self.gaussians, self.pipeline, self.background)
            # render_image = (render_result['render'].permute(1, 2, 0) * 255).to(torch.uint8).detach().cpu()
            render_image = self.get_render_image(current_camera)
            numpy_image = render_image.numpy()
            cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            return cv2_image
        except Exception as e:
            print(f"Error opencv: {e}")
            return None
    def get_rgba(self, image_path='train.png', current_camera = {'rot':[1, 0,0], 'pos': [1,1,1], 'width' : 1920, 'height': 1080 }):
        # img = cv2.imread(image_path)
        width, height = current_camera['width'], current_camera['height']
        img = self.get_render_cv2_image(current_camera)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        rgba = np.ones((height, width, 4), dtype=np.uint8) * 128
        """RGBA image buffer. The shape is (H, W, 4), following the NumPy convention."""
        rgba[:,:,3] = 255
        rgba[:,:,:3] = img * 255
        rgba_list =  rgba.flatten().tolist()
        return rgba_list

GS = GaussianSplattingServer()
GS.get_rgba()

from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2
import io
app = Flask(__name__)

@app.route('/get_rgba', methods=['POST'])
def api_get_rgba():
    """
    接收 POST 请求，输入为 JSON 格式的 current_camera 参数，返回 RGBA 矩阵。
    """
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # 检查必要参数是否存在
        required_keys = ['rot', 'pos', 'width', 'height']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required parameter: {key}"}), 400
        
        # 调用渲染函数生成 RGBA list
        rgba = GS.get_rgba(data)
        
        return jsonify({"rgba": rgba})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

import logging as _log

@app.route('/get_render', methods=['POST'])
def api_get_render():
    """
    接收 POST 请求，输入为 JSON 格式的 current_camera 参数，返回 RGBA 矩阵。
    """
    try:
        # 获取请求中的 JSON 数据
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # 检查必要参数是否存在
        required_keys = ['rot', 'pos', 'width', 'height']
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required parameter: {key}"}), 400
        
        cv2_image = GS.get_render_cv2_image(current_camera=data)
        success, jpeg_image = cv2.imencode('.jpg', cv2_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        # return send_file(io.BytesIO(cv2_image.tobytes()), mimetype='image/png')
        _log.warning("sucess render post")
        return send_file(io.BytesIO(jpeg_image.tobytes()), mimetype='image/jpeg')
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port='8892')