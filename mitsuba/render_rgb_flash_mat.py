"""
install mitsuba docker
https://github.com/benjamin-heasly/mitsuba-docker
"""
import os
import numpy as np
import json
import shutil
import imageio

imageio.plugins.freeimage.download()


asset_dir = 'synthetic_assets'
out_dir = 'output_folder'

light_offset = np.array([0.2, 0.1, 0.0])

for scene in os.listdir(asset_dir):
    in_scene_dir = os.path.join(asset_dir, scene)
    out_scene_dir = os.path.join(out_dir, scene)
    
    if not os.path.isdir(in_scene_dir):
        continue

    os.makedirs(out_scene_dir, exist_ok=True)
    
    light = 20.
    with open(os.path.join(out_scene_dir, 'light.txt'), 'w') as fp:
        fp.write(f'{light}\n')
    
    with open(os.path.join(out_scene_dir, 'light_offset.txt'), 'w') as fp:
        fp.write(f'{light_offset[0]} {light_offset[1]} {light_offset[2]}\n')

    for split in ['train', 'test']:
        out_split_dir = os.path.join(out_scene_dir, split)
        os.makedirs(os.path.join(out_split_dir, 'image'), exist_ok=True)

        cam_dict_fpath = os.path.join(asset_dir, f'{split}_cam_dict_norm.json')
        shutil.copy2(cam_dict_fpath, os.path.join(out_split_dir, 'cam_dict_norm.json'))
        
        cam_dict = json.load(open(cam_dict_fpath))
        img_list = list(cam_dict.keys())
        img_list = sorted(img_list, key=lambda x: int(x[:-4]))

        use_docker = True

        for index, img_name in enumerate(img_list[:1]):
            mesh = os.path.join(in_scene_dir, "model.obj")
            d_albedo = os.path.join(in_scene_dir, "diffuse_albedo.exr")
            s_albedo = os.path.join(in_scene_dir, "specular_albedo.exr")
            s_roughness = os.path.join(in_scene_dir, "specular_roughness.exr")

            K = np.array(cam_dict[img_name]["K"]).reshape((4, 4))
            focal = K[0, 0]
            width, height = cam_dict[img_name]["img_size"]
            fov = np.rad2deg(np.arctan(width / 2.0 / focal) * 2.0)
            w2c = np.array(cam_dict[img_name]["W2C"]).reshape((4, 4))
            # check if unit aspect ratio
            assert np.isclose(K[0, 0] - K[1, 1], 0.0), f"{K[0,0]} != {K[1,1]}"

            c2w = np.linalg.inv(w2c)
            c2w[:3, :2] *= -1  # mitsuba camera coordinate system: x-->left, y-->up, z-->scene
            origin = c2w[:3, 3]

            rotation = c2w[:3, :3]
            light_offset_world = rotation @ light_offset
            light_position = origin + light_offset_world
            
            c2w_str = " ".join([str(x) for x in c2w.flatten().tolist()])

            out_fpath = os.path.join(out_split_dir, 'image', img_name[:-4] + ".exr")
            cmd = (
                'mitsuba -b 10 rgb_flash_hdr_mat.xml -D fov={} -D width={} -D height={} -D c2w="{}" '
                "-D mesh={} -D d_albedo={} -D s_albedo={}  -D s_roughness={} "
                "-D light={} "
                "-D px={} -D py={} -D pz={}  "
                "-o {} ".format(
                    fov,
                    width,
                    height,
                    c2w_str,
                    mesh,
                    d_albedo,
                    s_albedo,
                    s_roughness,
                    light,
                    light_position[0],
                    light_position[1], 
                    light_position[2],
                    out_fpath,
                )
            )

            if use_docker:
                docker_prefix = "docker run -w `pwd` --rm -v `pwd`:`pwd` -v /phoenix:/phoenix ninjaben/mitsuba-rgb "
                cmd = docker_prefix + cmd

            os.system(cmd)
            os.system("rm -f mitsuba.*.log")

            light_info = {
                "camera_position": origin.tolist(),
                "light_position": light_position.tolist(),
                "light_offset_local": light_offset.tolist(),
                "light_offset_world": light_offset_world.tolist(),
                "light_intensity": light
            }
            with open(os.path.join(out_split_dir, 'image', img_name[:-4] + "_light_info.json"), 'w') as f:
                json.dump(light_info, f, indent=2)

