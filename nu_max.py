from mock_SONG import *
import cv2
import glob

img_array = []

for nu_max_i in np.linspace(2000, 4000, 21):
    print(nu_max_i)

    filename = './Figure/nu_max/nu_max=%.0f.png' % nu_max_i

    generate_acf(nu_max=nu_max_i,
                 delta_nu=delta_nu0,
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

out = cv2.VideoWriter('./Video/nu_max.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
