import numpy as np
import open3d as o3d
import generate_cloud 


nuage1_np = generate_cloud.generate_cloud('SLAM/images_test/set_3_caillou/frame_0001_l.jpg', 'SLAM/images_test/set_3_caillou/frame_0001_r.jpg')
nuage2_np = generate_cloud.generate_cloud('SLAM/images_test/set_3_caillou/frame_0002_l.jpg', 'SLAM/images_test/set_3_caillou/frame_0002_r.jpg')

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(nuage1_np)
pcd1.paint_uniform_color([1, 0, 0]) 

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(nuage2_np)
pcd2.paint_uniform_color([0, 0, 1])

distance_threshold = 0.5 
trans_init = np.eye(4) 


reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd2, pcd1, distance_threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("fitness :", reg_p2p.fitness)
print("rmse :", reg_p2p.inlier_rmse)

print("\nmtx déplacement")
print(reg_p2p.transformation)

pcd2.transform(reg_p2p.transformation)
o3d.visualization.draw_geometries([pcd1, pcd2], 
                                  window_name="Alignement ICP (Rouge=T1, Bleu=T2)",
                                  width=1024, height=768)
