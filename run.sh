# 单张图像
# python plane_detection.py \
#     --image /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/images/frame_00001.jpg \
#     --normal /media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/stable_normal/normal_vis/frame_00001.png \
#     --output outputs/coffee_room/frame_00001 \
#    --debug

python plane_detection.py \
  --image /media/wlt/Data/dataset/PlanarGS_dataset/replica/office0/images \
  --normal /media/wlt/Data/dataset/PlanarGS_dataset/replica/office0/aligned_normals_vis_da3 \
  --output outputs/replica/office0_with_da3