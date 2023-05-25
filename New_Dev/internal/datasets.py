# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Different datasets implementation plus a general port for all the datasets."""
import json
import os
from os import path
import queue
import threading

from internal import math, utils  # pylint: disable=g-multiple-import
import jax
import numpy as np
from PIL import Image


import cv2


def load_dataset(split, train_dir, config):
  """Loads a split of a dataset using the data_loader specified by `config`."""
  dataset_dict = {
      'blender': Blender,
      'multicam': Multicam,
      'blender_video': Blender_video,
      'multicam_video': Multicam_video,
  }
  return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_to_ndc(origins, directions, focal, width, height, near=1.,
                   focaly=None):
  """Convert a set of rays to normalized device coordinates (NDC).

  Args:
    origins: np.ndarray(float32), [..., 3], world space ray origins.
    directions: np.ndarray(float32), [..., 3], world space ray directions.
    focal: float, focal length.
    width: int, image width in pixels.
    height: int, image height in pixels.
    near: float, near plane along the negative z axis.
    focaly: float, Focal for y axis (if None, equal to focal).

  Returns:
    origins_ndc: np.ndarray(float32), [..., 3].
    directions_ndc: np.ndarray(float32), [..., 3].

  This function assumes input rays should be mapped into the NDC space for a
  perspective projection pinhole camera, with identity extrinsic matrix (pose)
  and intrinsic parameters defined by inputs focal, width, and height.

  The near value specifies the near plane of the frustum, and the far plane is
  assumed to be infinity.

  The ray bundle for the identity pose camera will be remapped to parallel rays
  within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
  world space can be remapped as long as it has dz < 0; this allows us to share
  a common NDC space for "forward facing" scenes.

  Note that
      projection(origins + t * directions)
  will NOT be equal to
      origins_ndc + t * directions_ndc
  and that the directions_ndc are not unit length. Rather, directions_ndc is
  defined such that the valid near and far planes in NDC will be 0 and 1.

  See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
  """

  # Shift ray origins to near plane, such that oz = -near.
  # This makes the new near bound equal to 0.
  t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions

  dx, dy, dz = np.moveaxis(directions, -1, 0)
  ox, oy, oz = np.moveaxis(origins, -1, 0)

  fx = focal
  fy = focaly if (focaly is not None) else focal

  # Perspective projection into NDC for the t = 0 near points
  #     origins + 0 * directions
  origins_ndc = np.stack([
      -2. * fx / width * ox / oz, -2. * fy / height * oy / oz,
      -np.ones_like(oz)
  ],
                         axis=-1)

  # Perspective projection into NDC for the t = infinity far points
  #     origins + infinity * directions
  infinity_ndc = np.stack([
      -2. * fx / width * dx / dz, -2. * fy / height * dy / dz,
      np.ones_like(oz)
  ],
                          axis=-1)

  # directions_ndc points from origins_ndc to infinity_ndc
  directions_ndc = infinity_ndc - origins_ndc

  return origins_ndc, directions_ndc


def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  max_fn = lambda x: max(x, patch_size)
  out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
  img = cv2.resize(img, out_shape, mode)
  return img


def focus_pt_fn(poses):
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[Ellipsis, :1, :4].shape)
  return np.concatenate([p[Ellipsis, :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[Ellipsis, :3, :4]


def recenter_poses(poses):
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses)
  poses = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  return unpad_poses(poses)


def shift_origins(origins, directions, near=0.0):
  """Shift ray origins to near plane, such that oz = near."""
  t = (near - origins[Ellipsis, 2]) / directions[Ellipsis, 2]
  origins = origins + t[Ellipsis, None] * directions
  return origins


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def normalize(x):
  """Normalization helper function."""
  return x / np.linalg.norm(x)


def generate_spiral_path(poses, bounds, n_frames=120, n_rots=2, zrate=.5):
  """Calculates a forward facing spiral path for rendering."""
  # Find a reasonable 'focus depth' for this dataset as a weighted average
  # of near and far bounds in disparity space.
  close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
  dt = .75
  focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

  # Get radii for spiral path using 90th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), 90, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    lookat = cam2world @ [0, 0, -focal, 1.]
    z_axis = position - lookat
    render_poses.append(viewmatrix(z_axis, up, position))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_spiral_path_dtu(poses, n_frames=120, n_rots=2, zrate=.5, perc=60):
  """Calculates a forward facing spiral path for rendering for DTU."""

  # Get radii for spiral path using 60th percentile of camera positions.
  positions = poses[:, :3, 3]
  radii = np.percentile(np.abs(positions), perc, 0)
  radii = np.concatenate([radii, [1.]])

  # Generate poses for spiral path.
  render_poses = []
  cam2world = poses_avg(poses)
  up = poses[:, :3, 1].mean(0)
  z_axis = focus_pt_fn(poses)
  for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
    t = radii * [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
    position = cam2world @ t
    render_poses.append(viewmatrix(z_axis, up, position, True))
  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def generate_hemispherical_orbit(poses, n_frames=120):
  """Calculates a render path which orbits around the z-axis."""
  origins = poses[:, :3, 3]
  radius = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))

  # Assume that z-axis points up towards approximate camera hemisphere
  sin_phi = np.mean(origins[:, 2], axis=0) / radius
  cos_phi = np.sqrt(1 - sin_phi**2)
  render_poses = []

  up = np.array([0., 0., 1.])
  for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
    camorigin = radius * np.array(
        [cos_phi * np.cos(theta), cos_phi * np.sin(theta), sin_phi])
    render_poses.append(viewmatrix(camorigin, up, camorigin))

  render_poses = np.stack(render_poses, axis=0)
  return render_poses


def transform_poses_to_hemisphere(poses, bounds):
  """Transforms input poses to lie roughly on the upper unit hemisphere."""

  # Use linear algebra to solve for the nearest point to the set of lines
  # given by each camera's focal axis
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

  # Recenter poses around this point and such that the world space z-axis
  # points up toward the camera hemisphere (based on average camera origin)
  toward_cameras = origins[Ellipsis, 0].mean(0) - focus_pt
  arbitrary_dir = np.array([.1, .2, .3])
  cam2world = viewmatrix(toward_cameras, arbitrary_dir, focus_pt)
  poses_recentered = np.linalg.inv(pad_poses(cam2world)) @ pad_poses(poses)
  poses_recentered = poses_recentered[Ellipsis, :3, :4]

  # Rescale camera locations (and other metadata) such that average
  # squared distance to the origin is 1 (so cameras lie roughly unit sphere)
  origins = poses_recentered[:, :3, 3]
  avg_distance = np.sqrt(np.mean(np.sum(origins**2, axis=-1)))
  scale_factor = 1. / avg_distance
  poses_recentered[:, :3, 3] *= scale_factor
  bounds_recentered = bounds * scale_factor

  return poses_recentered, bounds_recentered


def subsample_patches(images, patch_size, batch_size, batching='all_images'):
  """Subsamples patches."""
  n_patches = batch_size // (patch_size ** 2)

  scale = np.random.randint(0, len(images))
  images = images[scale]

  if isinstance(images, np.ndarray):
    shape = images.shape
  else:
    shape = images.origins.shape

  # Sample images
  if batching == 'all_images':
    idx_img = np.random.randint(0, shape[0], size=(n_patches, 1))
  elif batching == 'single_image':
    idx_img = np.random.randint(0, shape[0])
    idx_img = np.full((n_patches, 1), idx_img, dtype=int)
  else:
    raise ValueError('Not supported batching type!')

  # Sample start locations
  x0 = np.random.randint(0, shape[2] - patch_size + 1, size=(n_patches, 1, 1))
  y0 = np.random.randint(0, shape[1] - patch_size + 1, size=(n_patches, 1, 1))
  xy0 = np.concatenate([x0, y0], axis=-1)
  patch_idx = xy0 + np.stack(
      np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
      axis=-1).reshape(1, -1, 2)

  # Subsample images
  if isinstance(images, np.ndarray):
    out = images[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
  else:
    out = utils.dataclass_map(
        lambda x: x[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(  # pylint: disable=g-long-lambda
            -1, x.shape[-1]), images)
  return out, np.ones((n_patches, 1), dtype=np.float32) * scale


def anneal_nearfar(d, it, near_final, far_final,
                   n_steps=2000, init_perc=0.2, mid_perc=0.5):
  """Anneals near and far plane."""
  mid = near_final + mid_perc * (far_final - near_final)

  near_init = mid + init_perc * (near_final - mid)
  far_init = mid + init_perc * (far_final - mid)

  weight = min(it * 1.0 / n_steps, 1.0)

  near_i = near_init + weight * (near_final - near_init)
  far_i = far_init + weight * (far_final - far_init)

  out_dict = {}
  for (k, v) in d.items():
    if 'rays' in k and isinstance(v, utils.Rays):
      ones = np.ones_like(v.origins[Ellipsis, :1])
      rays_out = utils.Rays(
          origins=v.origins, directions=v.directions,
          viewdirs=v.viewdirs, radii=v.radii,
          lossmult=v.lossmult, near=ones*near_i, far=ones*far_i)
      out_dict[k] = rays_out
    else:
      out_dict[k] = v
  return out_dict


def sample_recon_scale(image_list, dist='uniform_scale'):
  """Samples a scale factor for the reconstruction loss."""
  if dist == 'uniform_scale':
    idx = np.random.randint(len(image_list))
  elif dist == 'uniform_size':
    n_img = np.array([i.shape[0] for i in image_list], dtype=np.float32)
    probs = n_img / np.sum(n_img)
    idx = np.random.choice(np.arange(len(image_list)), size=(), p=probs)
  return idx


class Dataset(threading.Thread):
  """Dataset Base Class."""

  def __init__(self, split, data_dir, config):
    super(Dataset, self).__init__()


    # here are the variables that added for the video dataset
    self.time_frame_num = 0
    self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
    self.daemon = True
    self._path_videodir = config.video_dir
    self.start_frame = config.start_frame
    self.end_frame = config.end_frame
    self.render_frame = config.render_frame
    # end of the new added arguments

    print("................. video dir ", self._path_videodir)
    self.use_tiffs = config.use_tiffs
    self.load_disps = config.compute_disp_metrics
    self.load_normals = config.compute_normal_metrics
    self.load_random_rays = config.load_random_rays
    self.load_random_fullimage_rays = config.dietnerf_loss_mult != 0.0
    self.load_masks = ((config.dataset_loader == 'dtu') and (split == 'test')
                       and (not config.dtu_no_mask_eval)
                       and (not config.render_path))
    self.checkpointdir = config.checkpoint_dir

    self.split = split
    if config.dataset_loader == 'dtu':
      self.data_base_dir = data_dir
      data_dir = os.path.join(data_dir, config.dtu_scan)
    elif config.dataset_loader == 'llff':
      self.data_base_dir = data_dir
      data_dir = os.path.join(data_dir, config.llff_scan)
    elif config.dataset_loader == 'blender':
      self.data_base_dir = data_dir
      data_dir = os.path.join(data_dir, config.blender_scene)
    self.data_dir = data_dir
    self.near = config.near
    self.far = config.far
    self.near_origin = config.near_origin
    self.anneal_nearfar = config.anneal_nearfar
    self.anneal_nearfar_steps = config.anneal_nearfar_steps
    self.anneal_nearfar_perc = config.anneal_nearfar_perc
    self.anneal_mid_perc = config.anneal_mid_perc
    self.sample_reconscale_dist = config.sample_reconscale_dist

    if split == 'train':
      self._train_init(config)
    elif split == 'test' or split == 'path':
      self._test_init(config)
    else:
      raise ValueError(
          f'`split` should be \'train\' or \'test\', but is \'{split}\'.')
    self.batch_size = config.batch_size // jax.host_count()
    self.batch_size_random = config.batch_size_random // jax.host_count()
    print('Using following batch size', self.batch_size)
    self.patch_size = config.patch_size
    self.batching = config.batching
    self.batching_random = config.batching_random
    self.render_path = config.render_path
    self.render_train = config.render_train
    self.start()
    

  def __iter__(self):
    return self

  def __next__(self):
    """Get the next training batch or test example.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.get()
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def peek(self):
    """Peek at the next training batch or test example without dequeuing it.

    Returns:
      batch: dict, has 'rgb' and 'rays'.
    """
    x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
    if self.split == 'train':
      return utils.shard(x)
    else:
      return utils.to_device(x)

  def run(self):
    if self.split == 'train':
      next_func = self._next_train
    else:
      next_func = self._next_test
    while True:
      self.queue.put(next_func())
      # print("Queue size", self.queue.qsize())

  @property
  def size(self):
    return self.n_examples

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    self._generate_downsampled_images(config)
    self._generate_rays(config)
    self._generate_downsampled_rays(config)

    # Generate more rays / image patches for unobserved-view-based losses.
    if self.load_random_rays:
      self._generate_random_rays(config)
    if self.load_random_fullimage_rays:
      self._generate_random_fullimage_rays(config)
      self._load_renderings_featloss(config)

    self.it = 0
    self.images_noreshape = self.images[0]

    if config.batching == 'all_images':
      # flatten the ray and image dimension together.
      self.images = [i.reshape(-1, 3) for i in self.images]
      if self.load_disps:
        self.disp_images = self.disp_images.flatten()
      if self.load_normals:
        self.normal_images = self.normal_images.reshape([-1, 3])

      self.ray_noreshape = [self.rays]
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, r.shape[-1]]), i) for (i, res) in zip(
              self.rays, self.resolutions)]

  
    elif config.batching == 'single_image':
      print("image shape:",self.images_noreshape.shape)
      self.images = [i.reshape(
          [-1, r, 3]) for (i, r) in zip(self.images, self.resolutions)]
      print("shape after operation:",self.images[0].shape)
      if self.load_disps:
        self.disp_images = self.disp_images.reshape([-1, self.resolution])
      if self.load_normals:
        self.normal_images = self.normal_images.reshape(
            [-1, self.resolution, 3])
      
      self.ray_noreshape = [self.rays]
      # print("rays:",self.rays)    
      self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
          [-1, res, r.shape[-1]]), i) for (i, res) in  # pylint: disable=cell-var-from-loop
                   zip(self.rays, self.resolutions)]
      print("self.resolutions:", self.resolutions)
      # print("rays:",self.rays)   
    else:
      raise NotImplementedError(
          f'{config.batching} batching strategy is not implemented.')
    # print("rays:",self.rays)
    with open('/media/pleasework/Storage/regnerf/New_Dev/debug/pop-dict.txt', 'w') as f:
      f.write(str(self.rays))

  def _test_init(self, config):
    self._load_renderings(config)
    if self.load_masks:
      self._load_masks(config)
    self._generate_rays(config)
    self.it = 0

  def _next_train(self):
    """Sample next training batch."""

    self.it = self.it + 1
    # print("self.it:",self.it)
    return_dict = {}
    if self.batching == 'all_images':
      # sample scale
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      ray_indices = np.random.randint(0, self.rays[idxs].origins.shape[0],
                                      (self.batch_size,))
      return_dict['rgb'] = self.images[idxs][ray_indices]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[ray_indices],
                                                self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[ray_indices]

    elif self.batching == 'single_image':
      idxs = sample_recon_scale(self.images, self.sample_reconscale_dist)
      idxs = 0
      # print("idxs:",idxs)
      image_index = np.random.randint(0, self.n_examples, ())
      ray_indices = np.random.randint(0, self.rays[idxs].origins[0].shape[0],
                                      (self.batch_size,))
      # print("ray_indices:", ray_indices)
      # print("image index:", image_index)
      return_dict['rgb'] = self.images[idxs][image_index][ray_indices]
      return_dict['rays'] = utils.dataclass_map(
          lambda r: r[image_index][ray_indices], self.rays[idxs])
      if self.load_disps:
        return_dict['disps'] = self.disp_images[image_index][ray_indices]
      if self.load_normals:
        return_dict['normals'] = self.normal_images[image_index][ray_indices]
    else:
      raise NotImplementedError(
          f'{self.batching} batching strategy is not implemented.')

    if self.load_random_rays:
      return_dict['rays_random'], return_dict['rays_random_scale'] = (
          subsample_patches(self.random_rays, self.patch_size,
                            self.batch_size_random,
                            batching=self.batching_random))
      return_dict['rays_random2'], return_dict['rays_random2_scale'] = (
          subsample_patches(
              self.random_rays, self.patch_size, self.batch_size_random,
              batching=self.batching_random))
    if self.load_random_fullimage_rays:
      idx_img = np.random.randint(self.random_fullimage_rays.origins.shape[0])
      return_dict['rays_feat'] = utils.dataclass_map(
          lambda x: x[idx_img].reshape(-1, x.shape[-1]),
          self.random_fullimage_rays)
      idx_img = np.random.randint(self.images_feat.shape[0])
      return_dict['image_feat'] = self.images_feat[idx_img].reshape(-1, 3)

    if self.anneal_nearfar:
      return_dict = anneal_nearfar(return_dict, self.it, self.near, self.far,
                                   self.anneal_nearfar_steps,
                                   self.anneal_nearfar_perc,
                                   self.anneal_mid_perc)
    
    with open('/home/pleasework/Desktop/Neural-motion-capture/debug/dan-dict-random-rays.txt', 'w') as f:
      f.write(str(return_dict))

    return return_dict

  def _next_test(self):
    """Sample next test example."""

    return_dict = {}

    idx = self.it
    self.it = (self.it + 1) % self.n_examples

    if self.render_path:
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx],
                                                self.render_rays)
    else:
      return_dict['rgb'] = self.images[idx]
      return_dict['rays'] = utils.dataclass_map(lambda r: r[idx], self.rays)

    if self.load_masks:
      return_dict['mask'] = self.masks[idx]
    if self.load_disps:
      return_dict['disps'] = self.disp_images[idx]
    if self.load_normals:
      return_dict['normals'] = self.normal_images[idx]

    return return_dict

  def _generate_rays(self, config):

    print("using the generate rays from dataset classsssssssss")
    """Generating rays for all images."""
    del config  # Unused.
    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(self.width, dtype=np.float32),  # X-Axis (columns)
        np.arange(self.height, dtype=np.float32),  # Y-Axis (rows)
        indexing='xy')
    camera_dirs = np.stack(
        [(x - self.width * 0.5 + 0.5) / self.focal,
         -(y - self.height * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
        axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        lossmult=ones,
        radii=radii,
        near=ones * self.near,
        far=ones * self.far)
    self.render_rays = self.rays

    with open('/home/pleasework/Desktop/Neural-motion-capture/debug/regnerf-rays.txt', 'w') as f:
      f.write(str(self.rays))

  def _generate_random_poses(self, config):
    """Generates random poses."""
    if config.random_pose_type == 'allposes':
      random_poses = list(self.camtoworlds_all)
    elif config.random_pose_type == 'renderpath':
      def sample_on_sphere(n_samples, only_upper=True, radius=4.03112885717555):
        p = np.random.randn(n_samples, 3)
        if only_upper:
          p[:, -1] = abs(p[:, -1])
        p = p / np.linalg.norm(p, axis=-1, keepdims=True) * radius
        return p

      def create_look_at(eye, target=np.array([0, 0, 0]),
                         up=np.array([0, 0, 1]), dtype=np.float32):
        """Creates lookat matrix."""
        eye = eye.reshape(-1, 3).astype(dtype)
        target = target.reshape(-1, 3).astype(dtype)
        up = up.reshape(-1, 3).astype(dtype)

        def normalize_vec(x, eps=1e-9):
          return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

        forward = normalize_vec(target - eye)
        side = normalize_vec(np.cross(forward, up))
        up = normalize_vec(np.cross(side, forward))

        up = up * np.array([1., 1., 1.]).reshape(-1, 3)
        forward = forward * np.array([-1., -1., -1.]).reshape(-1, 3)

        rot = np.stack([side, up, forward], axis=-1).astype(dtype)
        return rot

      origins = sample_on_sphere(config.n_random_poses)
      rotations = create_look_at(origins)
      random_poses = np.concatenate([rotations, origins[:, :, None]], axis=-1)
    else:
      raise ValueError('Not supported random pose type.')
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generating rays for all images."""
    self._generate_random_poses(config)

    random_rays = []
    for sfactor in [2**i for i in range(config.random_scales_init,
                                        config.random_scales)]:
      w = self.width // sfactor
      h = self.height // sfactor
      f = self.focal / (sfactor * 1.0)
      x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32),  # X-Axis (columns)
          np.arange(h, dtype=np.float32),  # Y-Axis (rows)
          indexing='xy')
      camera_dirs = np.stack(
          [(x - w * 0.5 + 0.5) / f,
           -(y - h * 0.5 + 0.5) / f, -np.ones_like(x)],
          axis=-1)
      directions = ((camera_dirs[None, Ellipsis, None, :] *
                     self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
      origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                                directions.shape)
      viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

      # Distance from each unit-norm direction vector to its x-axis neighbor.
      dx = np.sqrt(
          np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
      dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
      # Cut the distance in half, multiply it to match the variance of a uniform
      # distribution the size of a pixel (1/12, see paper).
      radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

      ones = np.ones_like(origins[Ellipsis, :1])
      rays = utils.Rays(
          origins=origins,
          directions=directions,
          viewdirs=viewdirs,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
      random_rays.append(rays)
    self.random_rays = random_rays

  def _load_renderings_featloss(self, config):
    """Loades renderings for DietNeRF's feature loss."""
    images = self.images[0]
    res = config.dietnerf_loss_resolution
    images_feat = []
    for img in images:
      images_feat.append(cv2.resize(img, (res, res), cv2.INTER_AREA))
    self.images_feat = np.stack(images_feat)

  def _generate_random_fullimage_rays(self, config):
    """Generating random rays for full images."""
    self._generate_random_poses(config)

    width = config.dietnerf_loss_resolution
    height = config.dietnerf_loss_resolution
    f = self.focal / (self.width * 1.0 / width)

    x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
        np.arange(width, dtype=np.float32) + .5,
        np.arange(height, dtype=np.float32) + .5,
        indexing='xy')

    camera_dirs = np.stack([(x - width * 0.5 + 0.5) / f,
                            -(y - height * 0.5 + 0.5) / f,
                            -np.ones_like(x)], axis=-1)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                   self.random_poses[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(self.random_poses[:, None, None, :3, -1],
                              directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
    # Cut the distance in half, multiply it to match the variance of a uniform
    # distribution the size of a pixel (1/12, see paper).
    radii = dx[Ellipsis, None] * 2 / np.sqrt(12)

    ones = np.ones_like(origins[Ellipsis, :1])
    self.random_fullimage_rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)

  def _generate_downsampled_images(self, config):
    """Generating downsampled images."""
    images = []
    resolutions = []
    for sfactor in [2**i for i in range(config.recon_loss_scales)]:
      # print("single image shape:",self.images[0].shape)
      imgi = np.stack([downsample(i, sfactor) for i in self.images])
      images.append(imgi)
      # print("single image shape:",imgi.shape)
      resolutions.append(imgi.shape[1] * imgi.shape[2])

    self.images = images
    self.resolutions = resolutions

  def _generate_downsampled_rays(self, config):
    """Generating downsampled images."""
    rays, height, width, focal = self.rays, self.height, self.width, self.focal
    ray_list = [rays]
    for sfactor in [2**i for i in range(1, config.recon_loss_scales)]:
      self.height = height // sfactor
      self.width = width // sfactor
      self.focal = focal * 1.0 / sfactor
      self._generate_rays(config)
      ray_list.append(self.rays)
    self.height = height
    self.width = width
    self.focal = focal
    self.rays = ray_list


class Multicam(Dataset):
  """Multicam Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the Multicam dataset.')
    with utils.open_file(path.join(self.data_dir, 'metadata.json'), 'r') as fp:
      self.meta = json.load(fp)[self.split]
      print("self.split:", self.split)
    self.meta = {k: np.array(self.meta[k]) for k in self.meta}
    # Should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta.
    # print(self.meta)
    images = []

    for fbase in self.meta['file_path']:
      print('file path:', fbase)
      fname = os.path.join(self.data_dir, fbase)
      with utils.open_file(fname, 'rb') as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.
      if config.white_background:
        image = image[Ellipsis, :3] * image[Ellipsis, -1:] + (1. - image[Ellipsis, -1:])
      images.append(image[Ellipsis, :3])

    self.images = np.stack(images, axis = 0)
    print("image shape after stack:", self.images.shape)
    self.n_examples = len(self.images)

    ## after the image read and the meta data read, now we are suppose to have pix2cam, cam2world, 
    ## width heiht and images

  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    # self._generate_downsampled_images(config)
    self._generate_rays(config)
    # self._generate_downsampled_rays(config)
    
    # To do, Zhenhui Lin 
    #  add the random full image ray and random rays later

    if self.load_random_rays:
      self._generate_random_rays(config)
    # if self.load_random_fullimage_rays:
    #   self._generate_random_fullimage_rays(config)
    #   self._load_renderings_featloss(config)


    self.it = 0
    self.images_noreshape = self.images[0]

    print("-----using multicam init----")

    def flatten(x):
      # Always flatten out the height x width dimensions
      x = [y.reshape([-1, y.shape[-1]]) for y in x]
      if config.batching == 'all_images':
        # If global batching, also concatenate all data into one list
        x = np.concatenate(x, axis=0)
      return x
    # self.images = np.stack(self.images, axis=0)
    self.images = flatten(self.images)
    self.images = np.stack(self.images, axis=0)
    print("after stack:",self.images.shape)
    self.images = [self.images]

    self.ray_noreshape = [self.rays]
    # print("rays:",self.rays)   
    self.rays = [self.rays] 
    self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
        [-1, res, r.shape[-1]]), i) for (i, res) in  # pylint: disable=cell-var-from-loop
                  zip(self.rays, self.resolutions)]
    # print("rays:",self.rays)  

  def _test_init(self, config):
    self._load_renderings(config)
    self._generate_rays(config)


  def _generate_rays(self, config):

    print("generate from the multicam loader")
    """Generating rays for all images."""
    pix2cam = self.meta['pix2cam']
    cam2world = self.meta['cam2world']
    width = self.meta['width']
    height = self.meta['height']
    self.resolutions = width * height

    def res2grid(w, h):
      return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
          np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
          indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]

    print("pixel_dirs shape:",pixel_dirs[0].shape, "pixcel_dir len:", len(pixel_dirs))

    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    camera_dirs = np.stack(camera_dirs, axis=0)

    directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
    directions = [-d[:, ::-1] for d in directions]
    directions = np.stack(directions, axis=0)
    origins = [
        np.broadcast_to(c2w[:3, -1], v.shape)
        for v, c2w in zip(directions, cam2world)
    ]
    origins = np.stack(origins, axis=0)
    viewdirs = [
        v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
    ]
    viewdirs = np.stack(viewdirs,axis=0)

    def broadcast_scalar_attribute(x):
      return [
          np.broadcast_to(x[i], origins[i][Ellipsis, :1].shape)
          for i in range(self.n_examples)
      ]

    lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
    near = broadcast_scalar_attribute(self.meta['near'])
    far = broadcast_scalar_attribute(self.meta['far'])
    ones = np.ones_like(origins[Ellipsis, :1])
    self.near = config.near
    self.far = config.far
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1)) for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], axis=0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[Ellipsis, None] * 2 / np.sqrt(12) for v in dx]
    radii = np.stack(radii,axis=0)

    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)
    
    self.camtoworlds_all = camera_dirs
    print("self.camtoworld_all shape:",self.camtoworlds_all.shape)
    self.bounds = np.stack([near, far], axis=-1)
    
    with open('/media/pleasework/Storage/regnerf/New_Dev/debug/regnerf_rays_dan_multicam.txt', 'w') as f:

      f.write(str(self.rays))

  def _generate_random_poses(self, config):
    """Generates random poses."""
    n_poses = config.n_random_poses
    cam2world = self.meta['cam2world']
    # print(cam2world)
    poses = np.stack(cam2world)
    print("pose shape:",poses.shape)
    bounds = self.bounds

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    print(positions)
    radii = np.percentile(np.abs(positions), 100, 0)
    print(radii)

    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
      t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal, 1.]
      z_axis = position - lookat
    
    random_poses.append(viewmatrix(z_axis, up, position))
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generates random rays."""
    self.n_examples = len(self.images)
    self._generate_random_poses(config)
    camtoworlds = self.random_poses
    pix2cam = self.meta['pix2cam']

    width = self.meta['width']
    height = self.meta['height']
    self.resolutions = width * height

    def res2grid(w, h):
      return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
          np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
          indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]

    print("pixel_dirs shape:",pixel_dirs[0].shape, "pixcel_dir len:", len(pixel_dirs))

    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    camera_dirs = np.stack(camera_dirs, axis=0)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                     camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                                directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # def broadcast_scalar_attribute(x):
    #   return [
    #       np.broadcast_to(x[i], origins[i][Ellipsis, :1].shape)
    #       for i in range(self.n_examples)
    #   ]

    # lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
    # near = broadcast_scalar_attribute(self.meta['near'])
    # far = broadcast_scalar_attribute(self.meta['far'])
    ones = np.ones_like(origins[Ellipsis, :1])
    self.near = 2
    self.far = 6
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1)) for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], axis=0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[Ellipsis, None] * 2 / np.sqrt(12) for v in dx]
    radii = np.stack(radii,axis=0)

    self.random_rays = utils.Rays(
        origins=origins[0],
        directions=directions[0],
        viewdirs=viewdirs[0],
        radii=radii[0],
        lossmult=ones[0],
        near=ones[0] * self.near,
        far=ones[0] * self.far)
    self.random_rays = [self.random_rays] 
    with open('/home/pleasework/Desktop/Neural-motion-capture/debug/random_rays_dan_multicam.txt', 'w') as f:

      f.write(str(self.random_rays))


class Blender(Dataset):
  """Blender Dataset."""

  def _load_renderings(self, config):
    """Load images from disk."""
    if config.render_path:
      raise ValueError('render_path cannot be used for the blender dataset.')
    print("self.data_dir:",self.data_dir)
    with utils.open_file(
        path.join(self.data_dir, f'transforms_{self.split}.json'), 'r') as fp:
      meta = json.load(fp)
    images = []
    disp_images = []
    normal_images = []
    cams = []
    for frame in meta['frames']:
      fprefix = os.path.join(self.data_dir, frame['file_path'])
      if self.use_tiffs:
        channels = []
        for ch in ['R', 'G', 'B', 'A']:
          with utils.open_file(fprefix + f'_{ch}.tiff', 'rb') as imgin:
            channels.append(np.array(Image.open(imgin), dtype=np.float32))
        # Convert image to sRGB color space.
        image = math.linear_to_srgb(np.stack(channels, axis=-1))
      else:
        with utils.open_file(fprefix + '.png', 'rb') as imgin:
          image = np.array(Image.open(imgin), dtype=np.float32) / 255.

      if self.load_disps:
        with utils.open_file(fprefix + '_disp.tiff', 'rb') as imgin:
          disp_image = np.array(Image.open(imgin), dtype=np.float32)
      if self.load_normals:
        with utils.open_file(fprefix + '_normal.png', 'rb') as imgin:
          normal_image = np.array(
              Image.open(imgin), dtype=np.float32)[Ellipsis, :3] * 2. / 255. - 1.

      if config.factor > 1:
        image = downsample(image, config.factor)
        if self.load_disps:
          disp_image = downsample(disp_image, config.factor)
        if self.load_normals:
          normal_image = downsample(normal_image, config.factor)

      cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
      images.append(image)
      if self.load_disps:
        disp_images.append(disp_image)
      if self.load_normals:
        normal_images.append(normal_image)

    self.images = np.stack(images, axis=0)
    if self.load_disps:
      self.disp_images = np.stack(disp_images, axis=0)
    if self.load_normals:
      self.normal_images = np.stack(normal_images, axis=0)

    rgb, alpha = self.images[Ellipsis, :3], self.images[Ellipsis, -1:]
    images = rgb * alpha + (1. - alpha) if config.white_background else rgb

    self.images_all = images
    self.camtoworlds_all = np.stack(cams, axis=0)
    if self.split == 'train' and config.n_input_views > 0:
      self.images = images[:config.n_input_views]
      self.camtoworlds = np.stack(cams[:config.n_input_views], axis=0)
    else:
      self.images = images
      self.camtoworlds = np.stack(cams, axis=0)

    self.height, self.width = self.images.shape[1:3]
    # self.height = self.height + 400
    self.resolution = self.height * self.width

    self.focal = .5 * self.width / np.tan(.5 * float(meta['camera_angle_x']))
    print("self.focal:",self.focal, "self.camera_angle_x:",meta['camera_angle_x'])
    self.n_examples = self.images.shape[0]


def write_meta(meta, config, save_meta=True):
  if save_meta and jax.host_id() == 0:
    os.makedirs(config.checkpoint_dir)
    with open(config.checkpoint_dir + '/meta.txt', 'w') as f:
      # write the meta txt file
      f.write(json.dumps(meta, indent=4))



class Multicam_video(Dataset):

  def _load_renderings(self, config):
      videos = []
      cap = cv2.VideoCapture(os.path.join(self._path_videodir, f'cam_1.mp4'))
      self.time_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      print("self.time_frame_num:", self.time_frame_num)
      print("we have {self.time_frame_num} frames in total for each videos from each camera.")
      
      # read the frame we need from the render frame and train on this frame only, we can later extend 
      # to the whole video by using a single loop


      if config.render_path:
          raise ValueError('render_path not supported for multicam video dataset.')
      
      # read the meta data:
      with utils.open_file(os.path.join(self.data_dir, 'meta.json'), 'r') as f:
          print("Now we are loading the metadata from split:", self.split)
          self.meta = json.load(f)[self.split]

      self.meta = {k: np.array(self.meta[k]) for k in self.meta}

      # get the number of cameras:

      pix2cam = self.meta['pix2cam']
      cam_num = pix2cam.shape[0]
      print("cam_num:", cam_num)

      img_all = []
      

      # for i in range(self.time_frame_num):
      for frame_idx in range(self.start_frame, self.end_frame):
        img_temp = []
        # print("frame idx:", frame_idx)

      
        for cam_idx in range(cam_num):
          cam_temp = cv2.VideoCapture(os.path.join(self._path_videodir, f'cam_{cam_idx+1}.mp4'))
          # print("cap", cam_idx+1, "generated!")
          # read the frame from the video and append to the img_temp
          cam_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
          ret, frame = cam_temp.read()
  
          if config.white_background:
            frame = frame[Ellipsis, :3] * frame[Ellipsis, -1:] + (1. - frame[Ellipsis, -1:])

          img_temp.append(frame)
          cv2.imwrite(os.path.join(self.checkpointdir, f'cam_{cam_idx+1}_{frame_idx}.png'), frame)
          
        images_per_frame = np.stack(img_temp, axis=0)
        # print("frame shape:", frame.shape, "img_temp len:", len(img_temp))
        img_all.append(images_per_frame)
        
      print("image all len and  shapes:", len(img_all), img_all[0].shape)

      self.images = img_all[0]
      print("self image shape: ",self.images.shape)
      self.n_examples = len(self.images)

      # print("self.images final shape:", self.images.shape)

      # lets check if we just get one frame from each camera and try to train it, is ok???



      # write the self.meta into the out path for saving the meta data, check if it is correct
      # or not???? 

      with utils.open_file(os.path.join(self.checkpointdir, 'metadata.txt'), 'w') as f:
          f.write(str(self.meta))
          

      # now we need to construct the time idx tensor for the model:
      self.time_idx = np.arange(self.time_frame_num)
      print("self.time_idx:", self.time_idx)
      self.time_idx = np.tile(self.time_idx, (cam_num, 1)).transpose()
      np.savetxt(os.path.join(self.checkpointdir, 'time_idx.txt'), self.time_idx, fmt='%d')
      # print("self.time_idx:", self.time_idx)

      # now we have the time serires, then we need to broadcast to the same shape as the near and fars,
      # but note that the time series will go through the mlp for predicting the color and density
      # thi should also be addressed



      #To do, zhenhuilin, after the travel:

      # 1. try to save all the images array with 20 frames, generate ray accordingly and also
      # plug in the time


  # -------------------------------- start the ray generation --------------------------------
  def _train_init(self, config):
    """Initialize training."""
    self._load_renderings(config)
    # self._generate_downsampled_images(config)
    self._generate_rays(config)
    # self._generate_downsampled_rays(config)

    if self.load_random_rays:
      self._generate_random_rays(config)
    # if self.load_random_fullimage_rays:
    #   self._generate_random_fullimage_rays(config)
    #   self._load_renderings_featloss(config)
    with open('/media/pleasework/Storage/regnerf/New_Dev/debug/pop-dict.txt', 'w') as f:
      f.write(str(self.rays))

    self.it = 0
    self.images_noreshape = self.images[0]

    print("-----using multicam_video init----")

    def flatten(x):
      # Always flatten out the height x width dimensions
      x = [y.reshape([-1, y.shape[-1]]) for y in x]
      if config.batching == 'all_images':
        # If global batching, also concatenate all data into one list
        x = np.concatenate(x, axis=0)
      return x
    
    # self.images = np.stack(self.images, axis=0)
    self.images = flatten(self.images)
    self.images = np.stack(self.images, axis=0)
    # print("after stack:",self.images.shape)
    self.images = [self.images]

    self.ray_noreshape = [self.rays]
    # print("rays:",self.rays)   
    self.rays = [self.rays] 
    self.rays = [utils.dataclass_map(lambda r: r.reshape(  # pylint: disable=g-long-lambda
        [-1, res, r.shape[-1]]), i) for (i, res) in  # pylint: disable=cell-var-from-loop
                  zip(self.rays, self.resolutions)]
      # print("rays:",self.rays)  

  def _test_init(self, config):
    self._load_renderings(config)
    self._generate_rays(config)
    self.it = 0

  def _generate_rays(self, config):

    # print("generate from the multicam loader")
    """Generating rays for all images."""
    pix2cam = self.meta['pix2cam']
    cam2world = self.meta['cam2world']
    width = self.meta['width']
    height = self.meta['height']
    self.resolutions = width * height

    def res2grid(w, h):
      return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
          np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
          indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]

    # print("pixel_dirs shape:",pixel_dirs[0].shape, "pixcel_dir len:", len(pixel_dirs))

    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    camera_dirs = np.stack(camera_dirs, axis=0)

    directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
    directions = [-d[:, ::-1] for d in directions]
    directions = np.stack(directions, axis=0)
    origins = [
        np.broadcast_to(c2w[:3, -1], v.shape)
        for v, c2w in zip(directions, cam2world)
    ]
    origins = np.stack(origins, axis=0)
    viewdirs = [
        v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
    ]
    viewdirs = np.stack(viewdirs,axis=0)

    def broadcast_scalar_attribute(x):
      return [
          np.broadcast_to(x[i], origins[i][Ellipsis, :1].shape)
          for i in range(self.n_examples)
      ]

    lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
    near = broadcast_scalar_attribute(self.meta['near'])
    far = broadcast_scalar_attribute(self.meta['far'])
    ones = np.ones_like(origins[Ellipsis, :1])
    print("ones.shape:", ones.shape)
    self.near = config.near
    self.far = config.far
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1)) for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], axis=0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[Ellipsis, None] * 2 / np.sqrt(12) for v in dx]
    radii = np.stack(radii,axis=0)


    # those origins, directions, viewdirs, radii, lossmult, near, far should be idx independent for
    # every timeframe and also, idx, will check out later and add................

    self.rays = utils.Rays(
        origins=origins,
        directions=directions,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=ones,
        near=ones * self.near,
        far=ones * self.far)
    
    self.camtoworlds_all = camera_dirs
    # print("self.camtoworld_all shape:",self.camtoworlds_all.shape)
    self.bounds = np.stack([near, far], axis=-1)
    

  def _generate_random_poses(self, config):
    """Generates random poses."""
    n_poses = config.n_random_poses
    cam2world = self.meta['cam2world']
    # print(cam2world)
    poses = np.stack(cam2world)
    # print("pose shape:",poses.shape)
    bounds = self.bounds

    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path using 90th percentile of camera positions.
    positions = poses[:, :3, 3]
    # print(positions)
    radii = np.percentile(np.abs(positions), 100, 0)
    # print(radii)

    radii = np.concatenate([radii, [1.]])

    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
      t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal, 1.]
      z_axis = position - lookat
    
    random_poses.append(viewmatrix(z_axis, up, position))
    self.random_poses = np.stack(random_poses, axis=0)

  def _generate_random_rays(self, config):
    """Generates random rays."""
    self.n_examples = len(self.images)
    self._generate_random_poses(config)
    camtoworlds = self.random_poses
    pix2cam = self.meta['pix2cam']

    width = self.meta['width']
    height = self.meta['height']
    self.resolutions = width * height

    def res2grid(w, h):
      return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
          np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
          np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
          indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]

    # print("pixel_dirs shape:",pixel_dirs[0].shape, "pixcel_dir len:", len(pixel_dirs))

    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    camera_dirs = np.stack(camera_dirs, axis=0)
    directions = ((camera_dirs[None, Ellipsis, None, :] *
                     camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
    origins = np.broadcast_to(camtoworlds[:, None, None, :3, -1],
                                directions.shape)
    viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

    # def broadcast_scalar_attribute(x):
    #   return [
    #       np.broadcast_to(x[i], origins[i][Ellipsis, :1].shape)
    #       for i in range(self.n_examples)
    #   ]

    # lossmult = broadcast_scalar_attribute(self.meta['lossmult'])
    # near = broadcast_scalar_attribute(self.meta['near'])
    # far = broadcast_scalar_attribute(self.meta['far'])
    ones = np.ones_like(origins[Ellipsis, :1])
    self.near = 2
    self.far = 6
    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1)) for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], axis=0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[Ellipsis, None] * 2 / np.sqrt(12) for v in dx]
    radii = np.stack(radii,axis=0)

    self.random_rays = utils.Rays(
        origins=origins[0],
        directions=directions[0],
        viewdirs=viewdirs[0],
        radii=radii[0],
        lossmult=ones[0],
        near=ones[0] * self.near,
        far=ones[0] * self.far)
    self.random_rays = [self.random_rays] 




class Blender_video(Dataset):
    def _load_renderings(self, config):
        videos = []