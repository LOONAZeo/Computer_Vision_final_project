import torch
import numpy as np
from data_loader import dense_to_sparse
import bz2
import os

file_path = "/home/student/Desktop/Salmon/rangeimage/testdata/001.npy"
output_folder = "/home/student/Desktop/Salmon/rangeimage/"

img = np.load(file_path)

occupancy_map = np.where(img != 0, 1, 0)

img = torch.tensor(img)


# Unsqueeze the tensor twice to make it four-dimensional
img = img.unsqueeze(0).unsqueeze(1)

img = dense_to_sparse(img)

original_coords = img.C
original_coords = original_coords.cpu()

# Convert the coordinates to bytes
byte_data = original_coords.numpy().tobytes()
byte_data_2 = occupancy_map.tobytes()

# Compress the byte data using bzip2
compressed_data = bz2.compress(byte_data)
compressed_data_2 = bz2.compress(byte_data_2)

# Save the compressed data to a file in the output folder
output_file_path = os.path.join(output_folder, "001.bz2")
with open(output_file_path, 'wb') as f:
    f.write(compressed_data)
output_file_path = os.path.join(output_folder, "001_occu_map.bz2")
with open(output_file_path, 'wb') as f:
    f.write(compressed_data_2)

# Calculate the compression ratio
original_size = len(byte_data) * 8  # Size in bits
compressed_size = len(compressed_data) * 8  # Size in bits
compression_ratio = original_size / compressed_size

# Decompress the data
decompressed_data = bz2.decompress(compressed_data)

# Convert the decompressed data back to torch tensor
reconstructed_coords = torch.tensor(np.frombuffer(decompressed_data, dtype=np.int32).reshape(-1, 4))

# Check if the reconstructed coordinates are identical to the original
identical = torch.equal(original_coords, reconstructed_coords)

# Calculate the bit rate
bit_rate = compressed_size / (original_coords.numel())  # Assuming each float32 coordinate takes 32 bits

print("Compression ratio:", compression_ratio)
print("Bit rate:", bit_rate, "bits per coordinate")
print("Reconstructed coordinates identical to original:", identical)