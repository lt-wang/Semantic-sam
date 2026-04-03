import os

gpu_id=0
quiet = True

#replica数据集
scenes = ['office0','office1','office2','office3','office4','room0','room1','room2']
data_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/replica'
output_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/replica'


for id, scene in enumerate(scenes):

    # common_args = f"--debug"
    common_args = f""
    quiet_arg = "--quiet" 
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python plane_detection.py \
        --image {data_base_path}/{scene}/images \
        --normal {data_base_path}/{scene}/geomprior_dust3r/normal_vis \
        --output {output_base_path}/{scene}/planarprior_dust3r \
        {quiet_arg} \
        {common_args}'
    print(cmd)
    os.system(cmd)

print("------------------------finish replica------------------------------")
