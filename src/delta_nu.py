from mock_SONG import *
import cv2
import glob

img_array = []

for delta_nu_i in np.linspace(100, 170, 36):
    print(delta_nu_i)

    filename = './Figure/delta_nu/delta_nu=%.0f.png' % delta_nu_i

    generate_acf(nu_max=nu_max0,
                 delta_nu=delta_nu_i,
                 epsilon=epsilon0,
                 Q=Q0,
                 amp=amp0,
                 sig=sig0,
                 N=N0,
                 filename=filename)

    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('./Video/delta_nu.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()