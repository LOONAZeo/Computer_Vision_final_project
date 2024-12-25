import os
import glob
from dataprocess.inout_points import cylinder_voxelize, write_ply_ascii_geo

def generate_dataset(input_dir, output_dir, grid_resolution, sequence):
    ply_files = sorted(glob.glob(os.path.join(input_dir, '*.ply')))
    print(f'Number of PLY files: {len(ply_files)}')

    for ply_file in ply_files:
        # Get reconstructed point cloud
        cylinder_voxel = cylinder_voxelize(ply_file, grid_resolution, sequence)
        data_pc_name = f"{os.path.splitext(os.path.basename(ply_file))[0]}_{sequence}.ply"
        output_cylinder_path = os.path.join(output_dir, data_pc_name)
        write_ply_ascii_geo(output_cylinder_path, cylinder_voxel)

#Generate the training dataset
if __name__ == "__main__":
    # sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
    sequences = ['11']
    GRID_RESOLUTION = [105600, 10800, 24000]    #Highest resolution
    CROP = False
    # OUTPUT_DIR = '/media/dsp5283090/Data/SematicKITTI_dataset/cy_train/Highest_resolution'
    OUTPUT_DIR = '/media/dsp5283090/Data/SematicKITTI_dataset/cy_test/Highest_resolution/11'

    for i in range(0, len(sequences)):
        DATA_DIR_ORI = f'/media/dsp5283090/Data/SematicKITTI_dataset/dataset_ply/sequence/{sequences[i]}/velodyne/'
        DATA_DIR_CROP = '/home/dsp5283090/Desktop/Jeremy/Cylinder2PCGC/ply_crop/'
        DATA_DIR = DATA_DIR_CROP if CROP else DATA_DIR_ORI

        print(f'Data Directory: {DATA_DIR}')
        print(f'Output Directory: {OUTPUT_DIR}')

        generate_dataset(DATA_DIR, OUTPUT_DIR, GRID_RESOLUTION, sequences[i])
