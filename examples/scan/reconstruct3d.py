import os
import sys
import view3d
from view3d import structuredlight as sl, camera, viewer
import numpy as np
import time
import re

if os.path.exists("pointcloud.ply"):
    app = viewer.app()
    app.load_pointcloud("pointcloud.ply")
    app.run()
    exit()


def get(content, url):
    import requests
    from zipfile import ZipFile
    headers = {'User-Agent': 'Mozilla/5.0'}
    if not os.path.exists(content):
        print("Downloading", content)
        with requests.get(url, headers=headers, stream=True) as r:
            total_length = r.headers.get('content-length')
            with open(content + '.zip', 'wb') as f:
                if total_length is None:  # no content length header
                    f.write(r.content)
                else:
                    dl, total_length = 0, int(total_length)
                    for data in r.iter_content(chunk_size=total_length // 100):
                        dl += len(data)
                        done = int(50 * dl / total_length)
                        f.write(data)
                        print("\r[{}{}]".format('=' * done, ' ' * (50-done)),
                              end='', flush=True)
        with ZipFile(content + '.zip', 'r') as zipObj:
            zipObj.extractall()
        os.remove(content + '.zip')
        print(" done!")
        return content

birdies = ("https://dtudk-my.sharepoint.com/"
           + ":u:/g/personal/sorgre_win_dtu_dk/"
           + "ESqN7uVPnQNKqCGHgrXjh44BE5YP31ovRdzUyJx19lS74w"
           + "?e=mhoqJv&download=1")
calibration = ("https://dtudk-my.sharepoint.com/"
               + ":u:/g/personal/sorgre_win_dtu_dk/"
               + "EXlhTuE1yC9DtwCHkOwtV1EBHu33gva9HBt9c7Qk7FbM3Q"
               + "?e=wBRfSi&download=1")

birdies = get('birdies', birdies)
calibration = get('calibration', calibration)

N = 16  # primary phaseshift step count
M = 8  # cue phaseshift step count
wn = 40.0  # phaseshift wave-number in inverse projector pixels

calibration = 'calibration'
capture = 'birdies'


def frmsort(filenames):
    names = [os.path.split(f)[1] for f in filenames]
    indices = [int(f.split('_')[1].split('.')[0]) for f in names]
    return [filenames[i] for i in np.argsort(indices)]

frexp = ['frame0_[0-9]*.png', 'frame1_[0-9]*.png']

print("Processing calibration sequence")
first_start = time.time()
frames = np.stack([view3d.io.read_images(calibration, frexp[0],
                                         frmsort, color=False),
                   view3d.io.read_images(calibration, frexp[1],
                                         frmsort, color=False)])
print("Images loaded [s]", time.time() - first_start)
print("Calib images:", frames.shape)

start = time.time()
reference = camera.checkerboard((22, 13), 15, coarse=0.5)
points3D, pxls = reference.find_in_image(frames)
print("Find reference [s]", time.time() - start)
print("Checkerboard pxl count:", (~np.isnan(pxls[..., 0])).sum())

start = time.time()
*cameras, _, _ = camera.stereocalibrate(points3D[0], *pxls, frames.shape)
print("Calibrate cameras [s]", time.time() - start)
# print("Cameras:\n", cameras[0], "\n", cameras[1])

start = time.time()
*rectified, R0, R1 = camera.stereorectify(*cameras)
print("... rectify [s]", time.time() - start)

start = time.time()
maps = np.array([camera.undistort_rectify_map(c, R, nc)
                 for c, R, nc in zip(cameras, [R0, R1], rectified)])
maps = maps.swapaxes(0, 1)
print("... maps [s]", time.time() - start)

print("Done calibrating!! [s]", time.time() - first_start)

frexp = ['frames0_[0-9]*.png', 'frames1_[0-9]*.png']

print("Processing capture:")
first_start = time.time()
frames = np.stack([view3d.io.read_images(capture, frexp[0], frmsort),
                   view3d.io.read_images(capture, frexp[1], frmsort)])
print("... loaded frames [s]", time.time() - first_start)

color_frames = frames[:, 0]
start = time.time()
grayscale = view3d.grayscale(frames).astype(np.float32) / 255
print("... grayscale frames [s]", time.time() - start)
encoded = view3d.ndsplit(grayscale, (1, 1, N, M), axis=1)
# encoded = lit, dark, primary, cue  <---  frames sequence of capture
start = time.time()
ph, dph, mask = sl.phaseshift.decode_with_cue(*encoded, wn)
print("... decoded frames [s]", time.time() - start)
start = time.time()
ph = view3d.interpolate.bilinear(ph[..., None], *maps)[..., 0]
dph = view3d.interpolate.bilinear(dph, *maps)
mask = view3d.interpolate.bilinear(mask.astype('u1')[..., None], *maps)[..., 0]
mask = mask > 0.5
print("... remapped codes [s]", time.time() - start)

start = time.time()
pxls = sl.correspondence.map_columns(*ph, *mask)
print("... match frames [s]", time.time() - start)

if len(pxls) == 0:
    print("No points reconstructed! [s]", time.time() - first_start)
    quit()

start = time.time()
ij = pxls.astype('f4')
color_frames = view3d.interpolate.bilinear(color_frames, *maps)
colors = view3d.interpolate.bilinear(color_frames[0], *ij[0].T) / 2
colors += view3d.interpolate.bilinear(color_frames[1], *ij[1].T) / 2
print("... interpolate colors [s]", time.time() - start)
start = time.time()
linearization = sl.triangulate.linearization(*rectified)
points = sl.triangulate.linear(*pxls, *linearization)
print("... triangulate [s]", time.time() - start)
start = time.time()
normals, _ = sl.estimatenormals.from_depth(rectified[0], points, pxls[0])

start = time.time()
ply = view3d.pointcloud(points, colors, normals).writePLY("pointcloud")
print("... ply files exported [s]", time.time() - start)

print("Sequence processed!! [s]", time.time() - first_start)

app = viewer.app()
app.load_pointcloud(ply)
app.run()
