#!/usr/bin/env python

from __future__ import print_function

import argparse
import os.path as osp
from struct import *

import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

from pointmvsnet.utils.io import *
from PIL import Image
from plyfile import PlyData, PlyElement


def mkdir(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)

def read_camera_parameters(filename,scale,index,flag):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size

    intrinsics[:2, :] *= scale

    if (flag==0):
        intrinsics[0,2]-=index
    else:
        intrinsics[1,2]-=index
  
    return intrinsics, extrinsics


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

def read_score_file(filename):
    data=[]
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            scores = [float(x) for x in f.readline().rstrip().split()[2::2]]
            data.append(scores)
    return data

# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            data.append((ref_view, src_views))
    return data

# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
                                ):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    masks=[]
    for i in range(2,11):
        mask = np.logical_and(dist < i/4, relative_depth_diff < i/1300)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src

def filter_depth(scan_folder, out_folder, plyfilename, photo_threshold):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair/pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    score_data = read_score_file(pair_file)

    nviews = len(pair_data)
    
    ct2 = -1

    for ref_view, src_views in pair_data:

        ct2 += 1
        print(src_views)

        # load the reference image
        ref_img = read_img(os.path.join(scan_folder, '{:0>8}.jpg'.format(ref_view)))
        # load the estimated depth of the reference view
        ref_depth_est = read_pfm(os.path.join(scan_folder, '{:0>8}_flow3.pfm'.format(ref_view)))[0]

        import cv2

        # load the photometric mask of the reference view
        confidence = read_pfm(os.path.join(scan_folder, '{:0>8}_flow3_prob.pfm'.format(ref_view)))[0]

        scale=float(confidence.shape[0])/ref_img.shape[0]
        index=int((int(ref_img.shape[1]*scale)-confidence.shape[1])/2)
        flag=0
        if (confidence.shape[1]/ref_img.shape[1]>scale):
            scale=float(confidence.shape[1])/ref_img.shape[1]
            index=int((int(ref_img.shape[0]*scale)-confidence.shape[0])/2)
            flag=1

        #confidence=cv2.pyrUp(confidence)
        ref_img=cv2.resize(ref_img,(int(ref_img.shape[1]*scale),int(ref_img.shape[0]*scale)))
        if (flag==0):
            ref_img=ref_img[:,index:ref_img.shape[1]-index,:]
        else:
            ref_img=ref_img[index:ref_img.shape[0]-index,:,:]

        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(ref_view)),scale,index,flag)

        photo_mask = confidence > photo_threshold

        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []
        # compute the geometric mask
        geo_mask_sum = 0
        geo_mask_sums=[]
        n=1
        for src_view in src_views:
          n+=1
        ct = 0
        for src_view in src_views:
                ct = ct + 1
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = read_camera_parameters(
                    os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(src_view)),scale,index,flag)
                # the estimated depth of the source view
                src_depth_est = read_pfm(os.path.join(out_folder, scan_folder, '{:0>8}_flow3.pfm'.format(src_view)))[0]

                src_confidence = read_pfm(os.path.join(out_folder, scan_folder, '{:0>8}_flow3_prob.pfm'.format(src_view)))[0]

                masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                            ref_extrinsics,
                                                                                            src_depth_est,
                                                                                            src_intrinsics, src_extrinsics)

                if (ct==1):
                    for i in range(2,n):
                        geo_mask_sums.append(masks[i-2].astype(np.int32))
                else :
                    for i in range(2,n):
                        geo_mask_sums[i-2]+=masks[i-2].astype(np.int32)

                geo_mask_sum+=geo_mask.astype(np.int32)

                all_srcview_depth_ests.append(depth_reprojected)


        geo_mask=geo_mask_sum>=n

        for i in range (2,n):
            geo_mask=np.logical_or(geo_mask,geo_mask_sums[i-2]>=i)
            print(geo_mask.mean())

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)


        if (not isinstance(geo_mask, bool)):

            final_mask = np.logical_and(photo_mask, geo_mask)

            os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)

            save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
            save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(scan_folder, ref_view,
                                                                                        photo_mask.mean(),
                                                                                        geo_mask.mean(),
                                                                                        final_mask.mean()))


            height, width = depth_est_averaged.shape[:2]
            x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
            valid_points = final_mask
            print("valid_points", valid_points.mean())
            x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
            color = ref_img[:, :, :][valid_points]  
            xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                                np.vstack((x, y, np.ones_like(x))) * depth)
            xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                                  np.vstack((xyz_ref, np.ones_like(x))))[:3]
            vertexs.append(xyz_world.transpose((1, 0)))
            vertex_colors.append((color * 255).astype(np.uint8))


    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_folder', type=str,
                        default='/home/zr/PointMVSNet/data/dtu/Eval/')
    parser.add_argument('--fusibile_exe_path', type=str, default='/home/zr/fusibile/fusibile')
    parser.add_argument('--init_prob_threshold', type=float, default=0.2) 
    parser.add_argument('--flow_prob_threshold', type=float, default=0.1) 
    parser.add_argument('--disp_threshold', type=float, default=0.12) 
    parser.add_argument('--num_consistent', type=int, default=3)
    parser.add_argument("-n", '--name', type=str, default='flow3')
    parser.add_argument("-m", '--inter_mode', type=str, default='LANCZOS4')
    parser.add_argument("-f", '--depth_folder', type=str, default='dtu_wde3')
    args = parser.parse_args()

    eval_folder = args.eval_folder
    init_prob_threshold = args.init_prob_threshold
    flow_prob_threshold = args.flow_prob_threshold
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent
    name = args.name

    DEPTH_FOLDER = args.depth_folder

    out_point_folder = os.path.join(eval_folder, DEPTH_FOLDER, '{}_3ITER_{}_ip{}_fp{}_d{}_nc{}'
                                    .format(args.inter_mode, name, init_prob_threshold, flow_prob_threshold,
                                            disp_threshold, num_consistent))
    mkdir(out_point_folder)

    scene_list = ["scan1"]

    for scene in scene_list:
        scene_folder = osp.join(eval_folder, DEPTH_FOLDER, scene)
        if not osp.isdir(scene_folder):
            continue
        if scene[:4] != "scan":
            continue
        print("**** Fusion for {} ****".format(scene))

        point_folder = osp.join(out_point_folder, scene)

        # dyfusion
        photo_threshold=0.3
        filter_depth(scene_folder, out_point_folder, os.path.join(out_point_folder, scene + '.ply'), photo_threshold)

        cur_dirs = os.listdir(point_folder)
        filter_dirs = list(filter(lambda x:x.startswith("consistencyCheck"), cur_dirs))

        rename_cmd = "cp " + osp.join(point_folder, filter_dirs[0]) + "/final3d_model.ply {}/{}{}_{}.ply".format(
            out_point_folder, "PointMVS", "{:0>3d}".format(int(scene[4:7])), "l3"
        )

        print(rename_cmd)
        os.system(rename_cmd)
