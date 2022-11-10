from mock_SONG import *
import cv2
import glob

img_array = []

for epsilon_i in np.arange(30):
	print(epsilon_i)

	filename = './Figure/epsilon/epsilon=%d.png' % epsilon_i

	generate_acf(nu_max=nu_max0,
				 delta_nu=delta_nu0,
				 epsilon=epsilon_i,
				 Q=Q0,
				 amp=amp0,
				 sig=sig0,
				 N=N0,
				 filename=filename)

	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width, height)
	img_array.append(img)

out = cv2.VideoWriter('./Video/epsilon.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
	out.write(img_array[i])
out.release()

