import pyzed.sl as sl
import cv2
import numpy as np

import sys
import viewer as gl
import pyzed.sl as sl
import argparse

from polygon_draw import PolygonDrawer
import matplotlib.pyplot as plt

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
from lightglue import viz2d
import torch
from collections import deque
import matplotlib.pyplot as plt
from pdb import set_trace as bp

torch.set_grad_enabled(False)

def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")



def main():
    print("Running Depth Sensing sample ... Press 'Esc' to quit\nPress 's' to save the point cloud")

    init = sl.InitParameters(depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    parse_args(init)
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    # res = sl.Resolution()
    # res.width = 2208
    # res.height = 1242

    camera_model = zed.get_camera_information().camera_model
    res = zed.get_camera_information().camera_configuration.resolution

    # bp()
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, camera_model, res)

    viewer_rgb = gl.GLViewer()
    viewer_rgb.init(1, sys.argv, camera_model, res)

    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_zed = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)

    #Queue for lightglue
    S = 2 

    new_frame_counter = 0
    new_frame_req = S
    frame_buffer = deque(maxlen=S)
    curr_frame_count = 0
    frame_buffer = []


    while viewer.is_available() and viewer_rgb.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            curr_frame_count+=1

            if curr_frame_count ==1:
                extractor = SuperPoint(max_num_keypoints=2048).eval().to(opt.device)  # load the extractor
                matcher = LightGlue(features="superpoint").eval().to(opt.device)

            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            # Use get_data() to get the numpy array
            image_ocv = image_zed.get_data()
            frame_buffer.append(torch.Tensor(image_ocv).permute(2,0,1))
            # print(frame_buffer[0].shape)

            # print(f'{image_ocv.shape=}')
            # print(f'{type(image_ocv)}')
            # bp()
            # Display the left image from the numpy array
            cv2.imwrite('RGB stream.jpg',image_ocv)
            # viewer_rgb.update(image_ocv)


            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, res)
            viewer.updateData(point_cloud)
            # bp()
            # print(dir(point_cloud))
            point_cloud_data = point_cloud.get_data()
            point3D = point_cloud.get_value(33,33)
            # print(f'{point3D=}')
            # print(f'{point_cloud_data.shape=}')
            # bp()
            
            if len(frame_buffer)==S:
                print("##############################DISPLAY IMAGE##################")
                plt.imshow(frame_buffer[0][:3,:,:])
                bp()
                feats0 = extractor.extract(frame_buffer[0][:3,:,:].to(opt.device))
                feats1 = extractor.extract(frame_buffer[1][:3,:,:].to(opt.device))

                matches01 = matcher({"image0": feats0, "image1": feats1})

                feats0, feats1, matches01 = [
                    rbd(x) for x in [feats0, feats1, matches01]
                ]  # remove batch dimension

                kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
                # print(m_kpts0.shape)
                # print(m_kpts1.shape)

                # print(type(frame_buffer[0].permute(1,2,0).cpu().numpy().astype(np.uint8)))
                axes = viz2d.plot_images([frame_buffer[0].permute(1,2,0).cpu().numpy().astype(np.uint8), frame_buffer[1].permute(1,2,0).cpu().numpy().astype(np.uint8)])
                # print("set axes")
                viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
                # print("plotted matches")
                viz2d.add_text(0, f'Stop after {matches01["stop"]} layers')
             
                viz2d.save_plot("./test.png")

                # kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
                # viz2d.plot_images([frame_buffer[0].permute(1,2,0).cpu().numpy().astype(np.uint8), frame_buffer[1].permute(1,2,0).cpu().numpy().astype(np.uint8)])
                # viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=6)
                
                curr_frame_count-=1
                print(curr_frame_count)



            if(viewer.save_data == True):
                point_cloud_to_save = sl.Mat()
                zed.retrieve_measure(point_cloud_to_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                err = point_cloud_to_save.write('Pointcloud.ply')
                if(err == sl.ERROR_CODE.SUCCESS):
                    print("Current .ply file saving succeed")
                else:
                    print("Current .ply file failed")
                viewer.save_data = False
    viewer.exit()
    zed.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--device', type=str, help='GPU(cuda) or CPU(cpu)', default = 'cuda')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.iogl_viewer.p_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
# if __name__ == '__main__':

#     zed = sl.Camera()
#     point_cloud = sl.Mat()

#     zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

#     point3D = point_cloud.get_value(0, 0)
#     x = point3D[0]
#     print(type(x))
#     y = point3D[1]
#     z = point3D[2]
#     color = point3D[3]

    
