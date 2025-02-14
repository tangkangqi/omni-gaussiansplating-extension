import copy
import ipywidgets
import json
import company.hello.world.kaolin as kaolin
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision

# Gaussian splatting dependencies
from utils.graphics_utils import focal2fov
from utils.system_utils import searchForMaxIteration
from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera as GSCamera

def log_tensor(t, name, **kwargs):
    print(kaolin.utils.testing.tensor_info(t, name=name, **kwargs))

class PipelineParamsNoparse:
    """ Same as PipelineParams but without argument parser. """
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

        
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
                    gt_alpha_mask=None, image_name='fake', uid=0)


# model_path = 'output/fa41de0a-3'
model_path = '/app/output/09ac9260-2/'
gaussians = load_checkpoint(model_path)
pipeline = PipelineParamsNoparse()
background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
test_camera = try_load_camera(model_path)

def compute_cam_fov(intrinsics, axis='x'):
    # compute FOV from focal
    aspectScale = intrinsics.width / 2.0
    tanHalfAngle = aspectScale / (intrinsics.focal_x if axis == 'x' else intrinsics.focal_y).item()
    fov = np.arctan(tanHalfAngle) * 2
    return fov

def convert_kaolin_camera(kal_camera):
    """ Convert kaolin camera to GS camera. """
    R = kal_camera.extrinsics.R[0]
    R[1:3] = -R[1:3]
    T = kal_camera.extrinsics.t.squeeze()
    T[1:3] = -T[1:3]
    return GSCamera(colmap_id=0,
                    R=R.transpose(1, 0).cpu().numpy(), 
                    T=T.cpu().numpy(), 
                    FoVx=compute_cam_fov(kal_camera.intrinsics, 'x'), 
                    FoVy=compute_cam_fov(kal_camera.intrinsics, 'y'), 
                    image=torch.zeros((3, kal_camera.height, kal_camera.width)),  # fake 
                    gt_alpha_mask=None,
                    image_name='fake',
                    uid=0)

def convert_gs_camera(gs_camera):
    """ Convert GS camera to Kaolin camera. """
    view_mat = gs_camera.world_view_transform.transpose(1, 0)
    view_mat[1:3] = -view_mat[1:3]
    res = kaolin.render.camera.Camera.from_args(
        view_matrix=view_mat,
        width=gs_camera.image_width, height=gs_camera.image_height,
        fov=gs_camera.FoVx, device='cpu')
    return res

# Test that we can render the same thing
kal_cam = convert_gs_camera(test_camera)