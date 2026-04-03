import cv2
input_path = "/media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/stable_normal/normal_vis/frame_00106.png"
rgb_path = "/media/wlt/Data/dataset/PlanarGS_dataset/mushroom/coffee_room/images/frame_00106.jpg"
out_dir = "outputs/normal_edges"

img = cv2.imread(input_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

cv2.imwrite(f"{out_dir}/frame_00106_edges.png", edges)
print(f"Edges saved to {out_dir}/frame_00106_edges.png")
