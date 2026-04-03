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




import plotly.graph_objects as go

print("\nGénération de la vue 3D interactive dans le navigateur...")

# On extrait les coordonnées X,Y,Z des deux nuages
pts1 = np.asarray(pcd1.points)
pts2 = np.asarray(pcd2.points)

fig = go.Figure()

# On ajoute le Nuage 1 (Rouge)
fig.add_trace(go.Scatter3d(
    x=pts1[:, 0], y=pts1[:, 2], z=-pts1[:, 1], # On réorganise les axes pour faire joli
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.8),
    name='Nuage T1 (Référence)'
))

# On ajoute le Nuage 2 (Bleu) - Celui qui a été déplacé par l'ICP
fig.add_trace(go.Scatter3d(
    x=pts2[:, 0], y=pts2[:, 2], z=-pts2[:, 1],
    mode='markers',
    marker=dict(size=2, color='blue', opacity=0.8),
    name='Nuage T2 (Aligné)'
))

# Paramètres d'affichage
fig.update_layout(
    title="Alignement ICP (Utilise le trackpad pour zoomer/tourner !)",
    scene=dict(
        xaxis_title='X (Largeur)',
        yaxis_title='Z (Profondeur)',
        zaxis_title='Y (Hauteur)',
        aspectmode='data' # Garde les vraies proportions
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show() # Ça va ouvrir un nouvel onglet dans ton navigateur web !