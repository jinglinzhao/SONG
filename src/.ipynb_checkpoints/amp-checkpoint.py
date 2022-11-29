from mock_SONG import *
import cv2
import glob

img_array = []

for amp_i in np.linspace(0.1, 1, 20):
	print(amp_i)

	filename = './Figure/amp/amp=%.4f.png' % amp_i

	generate_acf(nu_max=nu_max0,
				 delta_nu=delta_nu0,
				 epsilon=epsilon0,
				 Q=Q0,
				 amp=amp_i,
				 sig=sig0,
				 N=N0,
				 filename=filename)

	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width, height)
	img_array.append(img)

out = cv2.VideoWriter('./Video/amp.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
	out.write(img_array[i])
out.release()
