import os
import numpy as np

def read_ply_ascii_geo(filedir):
    files = open(filedir)
    data = []
    for i, line in enumerate(files):
        wordslist = line.split(' ')
        try:
            line_values = []
            for i, v in enumerate(wordslist):
                if v == '\n': continue
                line_values.append(float(v))
        except ValueError:
            continue
        data.append(line_values)
    data = np.array(data)
    coords = data[:, 0:3].astype('float')

    return coords

def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm ' + filedir)
    f = open(filedir, 'a+')
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    coords = coords.astype('float')
    for p in coords:
        f.writelines([str(p[0]), ' ', str(p[1]), ' ', str(p[2]), '\n'])
    f.close()

    return

class RangeProjection(object):
    """project 3d point cloud to 2d data with range projection"""

    def __init__(
        self,
        fov_up=3,
        fov_down=-25,
        proj_w=2048,
        proj_h=64,
        fov_left=-180,
        fov_right=180,
    ):
        # check params
        assert (
            fov_up >= 0 and fov_down <= 0
        ), "require fov_up >= 0 and fov_down <= 0, while fov_up/fov_down is {}/{}".format(
            fov_up, fov_down
        )
        assert (
            fov_right >= 0 and fov_left <= 0
        ), "require fov_right >= 0 and fov_left <= 0, while fov_right/fov_left is {}/{}".format(
            fov_right, fov_left
        )

        # params of fov angeles
        self.fov_up = fov_up / 180.0 * np.pi
        self.fov_down = fov_down / 180.0 * np.pi
        self.fov_vert = abs(self.fov_up) + abs(self.fov_down)  # 0.4886

        self.fov_left = fov_left / 180.0 * np.pi
        self.fov_right = fov_right / 180.0 * np.pi
        self.fov_hori = abs(self.fov_left) + abs(self.fov_right)  # 1.5707

        # params of proj img size
        self.proj_w = proj_w
        self.proj_h = proj_h

        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray, depth: np.ndarray = None):
        self.cached_data = {}
        pointcloud = pointcloud[:, :3]
        # get depth of all points
        if depth is None:
            depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
        # get point cloud components
        x = pointcloud[:, 0]
        y = pointcloud[:, 1]
        z = pointcloud[:, 2]

        # get angles of all points
        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / depth)

        # get projection in image coords
        proj_x = (
            yaw + abs(self.fov_left)
        ) / self.fov_hori  # normalized in [0, 1] # evenly divide in horizon
        proj_y = (
            1.0 - (pitch + abs(self.fov_down)) / self.fov_vert
        )  # normalized in [0, 1]

        # scale to image size using angular resolution
        proj_x *= self.proj_w
        proj_y *= self.proj_h

        # round and clamp for use as index
        proj_x = np.maximum(np.minimum(self.proj_w - 1, np.floor(proj_x)), 0).astype(
            np.int32
        )

        proj_y = np.maximum(np.minimum(self.proj_h - 1, np.floor(proj_y)), 0).astype(
            np.int32
        )

        self.cached_data["uproj_x_idx"] = proj_x.copy()
        self.cached_data["uproj_y_idx"] = proj_y.copy()
        self.cached_data["uproj_depth"] = depth.copy()

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        # order = np.argsort(depth)
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        yaw = yaw[order]
        pitch = pitch[order]

        yaw_proj = np.full((self.proj_h, self.proj_w), 0, dtype=np.float64)
        yaw_proj[proj_y, proj_x] = yaw

        pitch_proj = np.full((self.proj_h, self.proj_w), 0, dtype=np.float64)
        pitch_proj[proj_y, proj_x] = pitch

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), 0, dtype=np.float32)
        proj_range2 = np.full((self.proj_h, self.proj_w), 0, dtype=np.float16)
        proj_range[proj_y, proj_x] = depth
        proj_range2[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), 0, dtype=np.float32
        )
        proj_idx = np.full((self.proj_h, self.proj_w), 0, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)


        proj_pointcloud[proj_y, proj_x] = pointcloud

        return proj_pointcloud, proj_range, proj_idx, proj_mask, self.proj_w, self.proj_h, proj_x, proj_y, yaw, pitch


def split_array_into_images(up_low, array, output_folder):
    num_rows, num_cols = array.shape
    image_size = 64

    if num_rows % image_size != 0 or num_cols % image_size != 0:
        raise ValueError("The array dimensions must be divisible by the image size.")

    num_images_x = num_cols // image_size
    num_images_y = num_rows // image_size

    images = np.empty((num_images_y, num_images_x, image_size, image_size), dtype=array.dtype)

    for y in range(num_images_y):
        for x in range(num_images_x):
            start_row = y * image_size
            end_row = (y + 1) * image_size
            start_col = x * image_size
            end_col = (x + 1) * image_size

            image = array[start_row:end_row, start_col:end_col]
            images[y, x, :, :] = image
            if up_low == 'low':
                npy_path = os.path.join(output_folder, str(x) + '.npy')
                np.save(npy_path, image)
            elif up_low == 'up':
                npy_path = os.path.join(output_folder, str(x+16) + '.npy')
                np.save(npy_path, image)
            else:
                npy_path = os.path.join(output_folder, str(x) + '.npy')
                np.save(npy_path, image)
    return images


def reproject_to_3d_from_range(proj_range, proj_idx, yaw_rad, pitch_rad):
    # Flatten the projection arrays to 1D
    proj_range_flat = proj_range.flatten()
    proj_idx_flat = proj_idx.flatten()

    # Get the number of pixels in the range image
    num_pixels = proj_range_flat.shape[0]

    # Initialize arrays for x, y, z coordinates
    x = np.zeros(num_pixels)
    y = np.zeros(num_pixels)
    z = np.zeros(num_pixels)

    # Reconstruct the 3D coordinates (x, y, z) for each pixel in the range image
    for i in range(num_pixels):
        index = proj_idx_flat[i]
        if index >= 0:
            x[i] = proj_range_flat[i] * np.cos(pitch_rad[index]) * np.cos(yaw_rad[index])
            y[i] = proj_range_flat[i] * np.cos(pitch_rad[index]) * np.sin(-yaw_rad[index])
            z[i] = proj_range_flat[i] * np.sin(pitch_rad[index])

    # Reshape the 1D arrays to the original image dimensions
    # x = x.reshape(proj_range.shape)
    # y = y.reshape(proj_range.shape)
    # z = z.reshape(proj_range.shape)
    re_pc = np.stack((x, y, z), axis=0).T
    return re_pc

def reprojection(proj_range, proj_idx, yaw_rad, pitch_rad, h=64, w=1024):
    # pitch & yaw
    pitch = (1 - proj_range / h) * (3 + abs(-25)) - abs(-25)
    yaw = -(2 * proj_range / w - 1) * 180
    pitch_flat = pitch.flatten()
    yaw_flat = yaw.flatten()

    # Flatten the projection arrays to 1D
    proj_range_flat = proj_range.flatten()
    proj_idx_flat = proj_idx.flatten()

    # Get the number of pixels in the range image
    num_pixels = proj_range_flat.shape[0]

    # Initialize arrays for x, y, z coordinates
    x = np.zeros(num_pixels)
    y = np.zeros(num_pixels)
    z = np.zeros(num_pixels)

    # Reconstruct the 3D coordinates (x, y, z) for each pixel in the range image
    for i in range(num_pixels):
        index = proj_idx_flat[i]
        if index >= 0:
            x[i] = proj_range_flat[i] * np.cos(pitch_flat[index]) * np.cos(yaw_flat[index])
            y[i] = proj_range_flat[i] * np.cos(pitch_flat[index]) * np.sin(-yaw_flat[index])
            z[i] = proj_range_flat[i] * np.sin(pitch_flat[index])

    # Reshape the 1D arrays to the original image dimensions
    # x = x.reshape(proj_range.shape)
    # y = y.reshape(proj_range.shape)
    # z = z.reshape(proj_range.shape)
    re_pc = np.stack((x, y, z), axis=0).T
    return re_pc

def repro(ri, num_rows, num_columns):
    # Define the number of rows and columns

    # Calculate the interval between each row
    interval_pitch = 28 / num_rows
    interval_yaw = 360 / num_columns

    # Create the (64, 1024) 2D matrix
    pitch = []
    yaw = []

    #ford pitch from doc
    # # column = [-0.461611, -0.451281, -0.440090, -0.430000, -0.418945, -0.408667,
    #           -0.398230, -0.388220, -0.377890, -0.367720, -0.357393, -0.347628,
    #           -0.337549, -0.327694, -0.317849, -0.308124, -0.298358, -0.289066,
    #           -0.279139, -0.269655, -0.260049, -0.250622, -0.241152, -0.231731,
    #           -0.222362, -0.213039, -0.203702, -0.194415, -0.185154, -0.175909,
    #           -0.166688, -0.157484, -0.149826, -0.143746, -0.137673, -0.131631,
    #           -0.125582, -0.119557, -0.113538, -0.107534, -0.101530, -0.095548,
    #           -0.089562, -0.083590, -0.077623, -0.071665, -0.065708, -0.059758,
    #           -0.053810, -0.047868, -0.041931, -0.035993, -0.030061, -0.024124,
    #           -0.018193, -0.012259, -0.006324, -0.000393, 0.005547, 0.011485,
    #           0.017431, 0.023376, 0.029328, 0.035285]
    # column.reverse()

    #kitti pitch from doc
    # column = [-0.4666684, -0.4563029, -0.4449633, -0.4315151, -0.4218651, -0.4132595,
    #         -0.4029764, -0.3904468, -0.3798195, -0.3685576, -0.3589512, -0.3507572,
    #         -0.3405217, -0.3313, -0.3198341, -0.3094134, -0.2973507, -0.2901863,
    #         -0.2803818, -0.271589, -0.2605273, -0.2503052, -0.2398711, -0.2316849,
    #         -0.2228844, -0.214703, -0.2052908, -0.193627, -0.1848311, -0.177465,
    #         -0.1686703, -0.1588495, -0.1532154, -0.1460595, -0.1415509, -0.1348145,
    #         -0.1278769, -0.1227682, -0.1170488, -0.1109167, -0.1043736, -0.0982288,
    #         -0.0918913, -0.0859637, -0.0802411, -0.0737, -0.0687923, -0.0618433,
    #         -0.0565284, -0.0501877, -0.0444652, -0.0385384, -0.0319981, -0.0260716,
    #         -0.0209601, -0.0140075, -0.0078749, -0.0015374, 0.0031673, 0.0101177,
    #         0.0152269, 0.0227913, 0.027493, 0.0338299]
    # column.reverse()

    for _ in range(num_columns):
        column = [3 - (i + 0.5) * interval_pitch for i in range(num_rows)]
        #column = [3 - i * interval_pitch for i in range(num_rows)]
        pitch.append(column)

    for _ in range(num_rows):
        #row = [-180 + i * interval_yaw for i in range(num_columns)]
        row = [-180 + (i + 0.5) * interval_yaw for i in range(num_columns)]
        yaw.append(row)

    # Convert the list of lists into a NumPy array (optional)
    pitch_rad = np.array(pitch).T / 180.0 * np.pi
    # pitch_rad = np.array(pitch).T
    yaw_rad = np.array(yaw) / 180.0 * np.pi

    ri_flat = ri.flatten()
    pitch_flat = pitch_rad.flatten()
    yaw_flat = yaw_rad.flatten()

    num_pixels = ri_flat.shape[0]
    # Initialize arrays for x, y, z coordinates
    x = np.zeros(num_pixels)
    y = np.zeros(num_pixels)
    z = np.zeros(num_pixels)

    for i in range(num_pixels):
        x[i] = ri_flat[i] * np.cos(pitch_flat[i]) * np.cos(yaw_flat[i])
        y[i] = ri_flat[i] * -np.cos(pitch_flat[i]) * np.sin(yaw_flat[i])
        z[i] = ri_flat[i] * np.sin(pitch_flat[i])

    re_pc = np.stack((x, y, z), axis=0).T
    return re_pc
