from mock_SONG import *
import cv2
import glob

img_array = []

for sig_i in np.linspace(100, 1000, 19):
	print(sig_i)

	filename = './Figure/sig/sig=%d.png' % sig_i

	generate_acf(nu_max=nu_max0,
				 delta_nu=delta_nu0,
				 epsilon=epsilon0,
				 Q=Q0,
				 amp=amp0,
				 sig=sig_i,
				 N=N0,
				 filename=filename)

	img = cv2.imread(filename)
	height, width, layers = img.shape
	size = (width, height)
	img_array.append(img)

out = cv2.VideoWriter('./Video/sig.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

for i in range(len(img_array)):
	out.write(img_array[i])
out.release()

