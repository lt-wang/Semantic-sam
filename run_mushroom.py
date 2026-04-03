import os

gpu_id=0
quiet = True


#mushroom数据集
scenes = ['classroom','coffee_room','honka','kokko',"vr_room"]
data_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/mushroom'
output_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/mushroom'
for id, scene in enumerate(scenes):
    # common_args = f"--debug"
    common_args = f""
    quiet_arg = "--quiet" if quiet else ""
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python plane_detection.py \
        --image {data_base_path}/{scene}/images \
        --normal {data_base_path}/{scene}/geomprior_dust3r/normal_vis \
        --output {output_base_path}/{scene}/planarprior_dust3r \
        {quiet_arg} \
        {common_args}'
    print(cmd)
    os.system(cmd)
    print("------------------------finish mushroom------------------------------")
#
