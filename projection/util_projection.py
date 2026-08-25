import torch
import os
import numpy as np
from PIL import Image
from projection.obj_io_my import load_objs_as_meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams
)

class MeshRenderer3D:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lights = AmbientLights(device=self.device, ambient_color=[[0.8, 0.8, 0.8]])
        self.raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size = 0)
        self.render_view_list = [(0, 0), (0, 90), (0, 180), (0, -90), (90, 0), (-90, 0)]
        self.blend_params = BlendParams(background_color=(1.0, 1.0, 1.0))
        
    def load_and_normalize_mesh(self, file_path):
        mesh = load_objs_as_meshes([file_path], device=self.device)
        verts = mesh.verts_packed()
        verts_normalized = self.normalize_verts(verts)
        return mesh.update_padded(new_verts_padded=verts_normalized.unsqueeze(0))

    def normalize_verts(self, verts):
        centroid = verts.mean(dim=0)
        verts = verts - centroid
        max_length = torch.max(torch.sqrt(torch.sum(verts ** 2, dim=1)))
        return verts / max_length

    def phong_normal_shading(self, meshes, fragments) -> torch.Tensor:
        faces = meshes.faces_packed()  # (F, 3)
        vertex_normals = meshes.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        ones = torch.ones_like(fragments.bary_coords)
        pixel_normals = interpolate_face_attributes(fragments.pix_to_face, ones, faces_normals)
        return pixel_normals

    def render_views(self, file_path, save_path):
        self.save_path = save_path
        mesh = self.load_and_normalize_mesh(file_path)
        for i, (elev, azim) in enumerate(self.render_view_list):
            R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
            shader = SoftPhongShader(device=self.device, cameras=cameras, lights=self.lights, blend_params=self.blend_params)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

            # RGB 
            images = renderer(meshes_world=mesh)
            self.save_image(images, f"rendered_view_{i}.png")

            # normal
            fragments = rasterizer(mesh)
            normals = self.phong_normal_shading(mesh, fragments).squeeze().cpu().numpy()
            background_mask = self.save_normal_map(normals, f"rendered_normal_view_{i}.png")
            
            # mask
            self.save_mask(background_mask, f"rendered_mask_view_{i}.png")
            
            
    def render_views_eval(self, file_path):
        mesh = self.load_and_normalize_mesh(file_path)
        image_list= []
        for i, (elev, azim) in enumerate(self.render_view_list):
            R, T = look_at_view_transform(dist=2, elev=elev, azim=azim)
            cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
            shader = SoftPhongShader(device=self.device, cameras=cameras, lights=self.lights, blend_params=self.blend_params)
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(meshes_world=mesh)
            image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
            image_list.append(pil_image)
        return image_list

    def save_image(self, images, filename):
        image = (images[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        pil_image.save(os.path.join(self.save_path, filename))

    
    def save_normal_map(self, normal, filename):
        normalized_normal = (normal - normal.min()) / (normal.max() - normal.min())
        normal_image_data = (normalized_normal * 255).astype(np.uint8)
        background_mask = (normal == 0).all(axis=-1)
        normal_image_data[background_mask] = [255, 255, 255]
        normal_image = Image.fromarray(normal_image_data)
        normal_image.save(os.path.join(self.save_path, filename))
        return background_mask

    def save_mask(self, mask, filename):
        mask_image = np.zeros(mask.shape, dtype=np.uint8)
        mask_image[~mask] = 255  # Foreground (mesh) regions in white
        mask_pil = Image.fromarray(mask_image)
        mask_pil.save(os.path.join(self.save_path, filename))