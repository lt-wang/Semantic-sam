import os

gpu_id=0
quiet = True
#scannetpp数据集
scenes = ['8b5caf3398','b20a261fdf','66c98f4a9b','88cf747085']
# scenes = ['8b5caf3398','b20a261fdf','66c98f4a9b']
# scenes = ['88cf747085']
data_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/scannetpp'
output_base_path='/media/wlt/Data/dataset/PlanarGS_dataset/scannetpp'
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
print("------------------------finish scannetpp------------------------------")
