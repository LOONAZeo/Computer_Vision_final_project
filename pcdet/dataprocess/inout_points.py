# Copyright (c) Nanjing University, Vision Lab.
# Last update: 2019.09.17

import numpy as np
import matplotlib.pyplot as plt
import os, time
import pandas as pd
import subprocess

rootdir = os.path.split(__file__)[0]
################### plyfile <--> points ####################
def load_ply_data(filename):
  '''
  load data from ply file.
  '''
  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.int32)#np.uint8
  # print(filename,'\n','length:',points.shape)
  f.close()

  return points

def load_lidar_ply_data(filename):
  '''
  load data from ply file.
  '''
  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.float32)#np.uint8
  # print(filename,'\n','length:',points.shape)
  f.close()

  return points

def write_ply_data(filename, points):
  '''
  write data to ply file.
  '''
  if os.path.exists(filename):
    os.system('rm '+filename)
  f = open(filename,'a+')
  #print('data.shape:',data.shape)
  f.writelines(['ply\n','format ascii 1.0\n'])
  f.write('element vertex '+str(points.shape[0])+'\n')
  f.writelines(['property float x\n','property float y\n','property float z\n'])
  f.write('end_header\n')
  for _, point in enumerate(points):
    f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
  f.close() 

  return

#################### plyfile <--> partition points ####################

def load_points(filename, cube_size=64, min_num=60):
  """Load point cloud & split to cubes.
  
  Args: point cloud file; voxel size; minimun number of points in a cube.

  Return: cube positions & points in each cube.
  """

  # load point clouds
  point_cloud = load_ply_data(filename)
  # partition point cloud to cubes.
  cubes = {}# {block start position, points in block}
  for _, point in enumerate(point_cloud):
    cube_index = tuple((point//cube_size).astype("int"))
    local_point = point % cube_size
    if not cube_index in cubes.keys():
      cubes[cube_index] = local_point
    else:
      cubes[cube_index] = np.vstack((cubes[cube_index] ,local_point))
  # filter by minimum number.
  k_del = []
  for _, k in enumerate(cubes.keys()):
    if cubes[k].shape[0] < min_num:
      k_del.append(k)
  for _, k in enumerate(k_del):
    del cubes[k]
  # get points and cube positions.
  cube_positions = np.array(list(cubes.keys()))
  set_points = []
  # orderd
  step = cube_positions.max() + 1
  cube_positions_n = cube_positions[:,0:1] + cube_positions[:,1:2]*step + cube_positions[:,2:3]*step*step
  cube_positions_n = np.sort(cube_positions_n, axis=0)
  x = cube_positions_n % step
  y = (cube_positions_n // step) % step
  z = cube_positions_n // step // step
  cube_positions_orderd = np.concatenate((x,y,z), -1)
  for _, k in enumerate(cube_positions_orderd):
    set_points.append(cubes[tuple(k)].astype("int16"))

  return set_points, cube_positions

#################new add cylinder#################
def cart2polar(input_xyz):

  # print(f'Cart min bound: {np.min(input_xyz[:, :], axis=0)}')
  # print(f'Cart max bound: {np.max(input_xyz[:, :], axis=0)}')

  rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
  phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])

  return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar):
  # print(input_xyz_polar.shape)

  x = input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])
  y = input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])
  return np.stack((x, y, input_xyz_polar[:, 2]), axis=1)

def getGridInd(input_pol, grid_res):
  max_bound = np.max(input_pol[:, :], axis=0)
  min_bound = np.min(input_pol[:, :], axis=0)

  crop_range = max_bound - min_bound

  cur_grid_resolution = np.array(grid_res)
  gridSize = crop_range / (cur_grid_resolution - 1)
  if (gridSize == 0).any(): print("Zero grid_size!")

  gridInd = (np.around((np.clip(input_pol, min_bound, max_bound) - min_bound) / gridSize)).astype(int)

  return gridInd, min_bound, max_bound, gridSize

def load_points_cylinder(filename, grid_size):
  # load point clouds
  point_cloud = load_lidar_ply_data(filename)

  xyz_pol = cart2polar(point_cloud)

  min_deg = np.min(np.array(xyz_pol[:, 1]))
  print(f'min_deg:{min_deg}')
  min_height = np.min(np.array(xyz_pol[:, -1]))
  print(f'min_height:{min_height}')
  ###TODO
  max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)  # 82.953
  min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)  # 1.359

  max_bound = np.max(xyz_pol[:, 1:], axis=0)  # [50., 3.14, 2.]
  min_bound = np.min(xyz_pol[:, 1:], axis=0)  # [0., -3.14, -4.]
  max_bound = np.concatenate(([max_bound_r], max_bound))
  min_bound = np.concatenate(([min_bound_r], min_bound))

  crop_range = max_bound - min_bound  # [50. 6.28, 6.]
  cur_grid_size = np.array(grid_size)
  # intervals = crop_range / (cur_grid_size)
  intervals = np.array([15, 3, 3])
  if (intervals == 0).any(): print("Zero interval!")

  cubes = {}  # {block start position, points in block}
  cubes_allpts = {}

  for _, point in enumerate(xyz_pol):
    cube_index = tuple((np.floor((np.clip(point, min_bound, max_bound) - min_bound) / intervals)).astype(int))
    local_point = np.zeros(3)
    local_point_float = np.zeros(3)
    if point[0] >= 0:
      local_point[0] = point[0] % intervals[0]
    else:
      local_point[0] = (point[0] % intervals[0]) - intervals[0]
    if point[1] >= 0:
      local_point[1] = point[1] % intervals[1]
    else:
      local_point[1] = (point[1] % intervals[1]) - intervals[1]
    if point[2] >= 0:
      local_point[2] = point[2] % intervals[2]
    else:
      local_point[2] = (point[2] % intervals[2]) - intervals[2]

    # local_point = point % intervals

    #for calculate all pts
    if point[0] >= 0:
      local_point_float[0] = point[0] % intervals[0]
    else:
      local_point_float[0] = (point[0] % intervals[0]) - intervals[0]
    if point[1] >= 0:
      local_point_float[1] = point[1] % intervals[1]
    else:
      local_point_float[1] = (point[1] % intervals[1]) - intervals[1]
    if point[2] >= 0:
      local_point_float[2] = point[2] % intervals[2]
    else:
      local_point_float[2] = (point[2] % intervals[2]) - intervals[2]

    local_point = local_point.astype('int16')
    if not cube_index in cubes.keys():
      cubes[cube_index] = np.atleast_2d(local_point)  #make sure it's 2 ndim
    else:
      if np.any(np.all(local_point == cubes[cube_index], axis=1)):
        pass
      else:
        cubes[cube_index] = np.vstack((cubes[cube_index], local_point))

    if not cube_index in cubes_allpts.keys():
      cubes_allpts[cube_index] = np.atleast_2d(local_point_float)  #make sure it's 2 ndim
    else:
      cubes_allpts[cube_index] = np.vstack((cubes_allpts[cube_index], local_point_float))

  # get points and cube positions.
  cube_positions = np.array(list(cubes.keys()))

  # Calculate the average number of values per key
  total_values = sum(len(v) for v in cubes.values())
  average_values = total_values / len(cubes)
  print(f'lencubes: {len(cubes)}')
  print(f"Average number of values per key in cubes: {average_values}")

  total_values = sum(len(v) for v in cubes_allpts.values())
  average_values = total_values / len(cubes_allpts)
  print(f'lencubes float: {len(cubes_allpts)}')
  print(f"Average number of values per key in cubes float: {average_values}")

  set_points = []
  # orderd
  step = cube_positions.max() + 1
  cube_positions_n = cube_positions[:, 0:1] + cube_positions[:, 1:2] * step + cube_positions[:, 2:3] * step * step
  cube_positions_n = np.sort(cube_positions_n, axis=0)
  x = cube_positions_n % step
  y = (cube_positions_n // step) % step
  z = cube_positions_n // step // step
  cube_positions_orderd = np.concatenate((x, y, z), -1)

  for _, k in enumerate(cube_positions_orderd):
    # set_points.append(cubes[tuple(k)])
    set_points.append(cubes[tuple(k)].astype("int16"))

  # calculate nonempty voxel %
  grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(int)
  nonempty_voxel = np.unique(grid_ind, axis=0)
  total_nonempty_voxel = len(nonempty_voxel)
  total_voxel = np.prod(cur_grid_size)
  persent_nonempty_voxel = (total_nonempty_voxel / total_voxel) * 100
  print(f'total voxels: {total_voxel}')
  print(f'nonempty voxel: {total_nonempty_voxel}')
  print(f'nonempty voxel(%): {persent_nonempty_voxel} %')

  # calculate loss points %
  total_points = len(xyz_pol)
  persent_loss_points = ((total_points - total_nonempty_voxel) / total_points) * 100
  print(f'total points: {total_points}')
  print(f'loss points(%): {persent_loss_points} %')
  print('-----------------------------------------------')

  print(f'type:{type(grid_ind)}, shape:{np.shape(grid_ind)}')
  print(f'type:{type(xyz_pol)}, shape:{np.shape(xyz_pol)}')

  return set_points, cube_positions

def write_ply_ascii_geo(filedir, coords):
  if os.path.exists(filedir): os.system('rm ' + filedir)
  f = open(filedir, 'a+')
  f.writelines(['ply\n', 'format ascii 1.0\n'])
  f.write('element vertex ' + str(coords.shape[0]) + '\n')
  f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
  f.write('end_header\n')
  # coords = coords.astype('int')
  for p in coords:
    f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), '\n'])
  f.close()

  return

def pc_error(infile1, infile2, res, normal=False, show=False):
  # Symmetric Metrics. D1 mse, D1 hausdorff.
  headers1 = ["mse1      (p2point)", "mse1,PSNR (p2point)",
              "h.       1(p2point)", "h.,PSNR  1(p2point)"]

  headers2 = ["mse2      (p2point)", "mse2,PSNR (p2point)",
              "h.       2(p2point)", "h.,PSNR  2(p2point)"]

  headersF = ["mseF      (p2point)", "mseF,PSNR (p2point)",
              "h.        (p2point)", "h.,PSNR   (p2point)"]

  haders_p2plane = ["mse1      (p2plane)", "mse1,PSNR (p2plane)",
                    "mse2      (p2plane)", "mse2,PSNR (p2plane)",
                    "mseF      (p2plane)", "mseF,PSNR (p2plane)"]

  headers = headers1 + headers2 + headersF + haders_p2plane

  command = str(rootdir + '/pc_error' +
                ' -a ' + infile1 +
                ' -b ' + infile2 +
                #   ' -n '+infile1+
                ' --hausdorff=1 ' +
                ' --resolution=' + str(res - 1))

  if normal:
    headers += haders_p2plane
    command = str(command + ' -n ' + infile1)

  results = {}

  start = time.time()
  subp = subprocess.Popen(command,
                          shell=True, stdout=subprocess.PIPE)

  c = subp.stdout.readline()
  while c:
    line = c.decode(encoding='utf-8')  # python3.
    if show:
      print(line)
    for _, key in enumerate(headers):
      if line.find(key) != -1:
        value = number_in_line(line)
        results[key] = value

    c = subp.stdout.readline()
    # print('===== measure PCC quality using `pc_error` version 0.13.4', round(time.time() - start, 4))

  return pd.DataFrame([results])

def cylinder_voxelize(filename, grid_resolution, sequence):
  data_pc = f"{os.path.splitext(os.path.basename(filename))[0]}_{sequence}.ply"
  print(f'------{data_pc}------')

  # Load point clouds
  point_cloud = load_lidar_ply_data(filename)
  xyz_pol = cart2polar(point_cloud)

  grid_ind, polar_min_bound, polar_max_bound, grid_size = getGridInd(xyz_pol, grid_resolution)
  unique_grid_ind, indices = np.unique(grid_ind, axis=0, return_index=True)
  unsorted_unique_grid_ind = unique_grid_ind[np.argsort(indices)]

  return unsorted_unique_grid_ind

def cylinder_voxelize_reconstruct(filename, grid_resolution):
  data_pc = f"{os.path.splitext(os.path.basename(filename))[0]}_re.ply"
  print(f'------{data_pc}------')

  # Load point clouds
  point_cloud = load_lidar_ply_data(filename)
  xyz_pol = cart2polar(point_cloud)

  grid_ind, polar_min_bound, polar_max_bound, grid_size = getGridInd(xyz_pol, grid_resolution)
  unique_grid_ind, indices = np.unique(grid_ind, axis=0, return_index=True)
  unsorted_unique_grid_ind = unique_grid_ind[np.argsort(indices)]
  inv_polar = unsorted_unique_grid_ind * grid_size + polar_min_bound

  inv_cat = polar2cat(inv_polar)

  print(f'Input points: {len(point_cloud)}')
  print(f'Output points: {len(inv_polar)}')
  print(f'Max bound: {polar_max_bound}')
  print(f'Min bound: {polar_min_bound}')
  print(f'Crop range: {polar_max_bound - polar_min_bound}')
  print(f'Grid resolution: {grid_resolution}')
  print(f'Grid size: {grid_size}')

  return inv_cat

def crop_ply(filename, max_bound, min_bound):
  point_cloud = load_lidar_ply_data(filename)

  xyz_pol = cart2polar(point_cloud)
  croped_pol = []

  for point in xyz_pol:
    if all(min_bound[i] <= point[i] <= max_bound[i] for i in range(len(point))):
      croped_pol.append(point)

  croped_pol = np.array(croped_pol)
  croped_pc = polar2cat(croped_pol)

  return croped_pc

def cal_density(filename, grid_resolution):
  # load point clouds
  point_cloud = load_lidar_ply_data(filename)

  xyz_pol = cart2polar(point_cloud)

  grid_ind, min_bound, max_bound, grid_size = getGridInd(xyz_pol, grid_resolution)
  unique_grid_ind = np.unique(grid_ind, axis=0)

  gridThreshold = []
  disPts = []
  disPts_voxel = []
  disRange = np.arange(5, 85, 5)
  for threshold in range(5, 80, 5):
    disPts.append(np.sum((xyz_pol[:, 0] <= threshold) & (xyz_pol[:, 0] > threshold - 5)))
    gridThreshold.append(np.floor((threshold - min_bound[0]) / grid_size[0]).astype(int))
  disPts.append(np.sum(xyz_pol[:, 0] > 75))

  for i, threshold in enumerate(gridThreshold):
    if i == 0:
      disPts_voxel.append(np.sum((unique_grid_ind[:, 0] <= threshold)))
    else:
      disPts_voxel.append(np.sum((unique_grid_ind[:, 0] <= threshold) & (unique_grid_ind[:, 0] > gridThreshold[i-1])))
  disPts_voxel.append(np.sum((unique_grid_ind[:, 0] > threshold)))

  loss_persent = 1 - (np.array(disPts_voxel)/np.array(disPts))

  plt.plot(disRange, disPts, c='tab:red', label='point')
  plt.plot(disRange, disPts_voxel, c='tab:blue', label='voxel')
  plt.title('LiDAR distribution')
  plt.ylabel('Number of points/voxels')
  plt.xlabel('Distance(m)')
  plt.legend(['point', 'voxel'], loc='upper right')
  plt.savefig("LiDAR_distribution.png")
  plt.close()

  return

#################new add cylinder end#################

def save_points(set_points, cube_positions, filename, cube_size=64):
  """Combine & save points."""

  # order cube positions.
  step = cube_positions.max() + 1
  cube_positions_n = cube_positions[:,0:1] + cube_positions[:,1:2]*step + cube_positions[:,2:3]*step*step
  cube_positions_n = np.sort(cube_positions_n, axis=0)
  x = cube_positions_n % step
  y = (cube_positions_n // step) % step
  z = cube_positions_n // step // step
  cube_positions_orderd = np.concatenate((x,y,z), -1)
  # combine points.
  point_cloud = []
  for k, v in zip(cube_positions_orderd, set_points):
    points = v + np.array(k) * cube_size
    point_cloud.append(points)
  point_cloud = np.concatenate(point_cloud).astype("int")
  
  write_ply_data(filename, point_cloud)

  return

#################### points <--> volumetric models ####################

def points2voxels(set_points, cube_size):
  """Transform points to voxels (binary occupancy map).
  Args: points list; cube size;

  Return: A tensor with shape [batch_size, cube_size, cube_size, cube_size, 1]
  """

  voxels = []
  for _, points in enumerate(set_points):
    points = points.astype("int")
    vol = np.zeros((cube_size,cube_size,cube_size))
    vol[points[:,0],points[:,1],points[:,2]] = 1.0
    vol = np.expand_dims(vol,-1)
    voxels.append(vol)
  voxels = np.array(voxels)

  return voxels

def voxels2points(voxels):
  """extract points from each voxel."""

  voxels = np.squeeze(np.uint8(voxels)) # 0 or 1
  set_points = []
  for _, vol in enumerate(voxels):
    points = np.array(np.where(vol>0)).transpose((1,0))
    set_points.append(points)
  
  return set_points

#################### select top-k voxels of volumetric model ####################

def select_voxels(vols, points_nums, offset_ratio=1.0, fixed_thres=None):
  '''Select the top k voxels and generate the mask.
  input:  vols: [batch_size, vsize, vsize, vsize, 1] float32
          points numbers: [batch_size]
  output: the mask (0 or 1) representing the selected voxels: [batch_size, vsize, vsize, vsize]  
  '''
  # vols = vols.numpy()
  # points_nums = points_nums
  # offset_ratio = offset_ratio
  masks = []
  for idx, vol in enumerate(vols):
    if fixed_thres==None:
      num = int(offset_ratio * np.array(points_nums[idx]))
      thres = get_adaptive_thres(vol, num)
    else:
      thres = fixed_thres
    # print(thres)
    # mask = np.greater(vol, thres).astype('float32')
    mask = np.greater_equal(vol, thres).astype('float32')
    masks.append(mask)

  return np.stack(masks)

def get_adaptive_thres(vol, num, init_thres=-2.0):
  values = vol[vol>init_thres]
  # number of values should be larger than expected number.
  if values.shape[0] < num:
    values = np.reshape(vol, [-1])
  # only sort the selected values.
  values.sort()
  thres = values[-num]

  return thres


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--input", type=str, default='../../testdata/8iVFB/redandblack_vox10_1550_n.ply', dest="input")
  parser.add_argument(
    "--output", type=str, default='rec.ply', dest="output")
  parser.add_argument(
    "--cube_size", type=int, default=64, dest="cube_size",
    help="size of partitioned cubes.")
  parser.add_argument(
    "--min_num", type=int, default=20, dest="min_num",
    help="minimum number of points in a cube.")
  args = parser.parse_args()
  print(args)

  ################### test top-k voxels selection #########################
  data = np.random.rand(4, 64, 64, 64, 1) * (100) -50
  points_nums = np.array([1000, 200, 10000, 50])
  offset_ratio = 1.0 
  init_thres = -1.0
  mask = select_voxels(data, points_nums, offset_ratio, init_thres)   
  print(mask.shape)

  ################### inout #########################
  set_points, cube_positions = load_points(args.input, 
                                          cube_size=args.cube_size, 
                                          min_num=args.min_num)
  voxels = points2voxels(set_points, cube_size=args.cube_size)
  print('voxels:',voxels.shape)
  points_rec = voxels2points(voxels)
  save_points(points_rec, cube_positions, args.output, cube_size=args.cube_size)
  os.system("../myutils/pc_error_d" \
  + ' -a ' + args.input + ' -b ' + args.output + " -r 1023")
  os.system("rm "+args.output)
