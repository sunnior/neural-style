

def get_img(src, img_size=False):
	img = scipy.misc.imread(src, mode='RGB') # misc.imresize(, (256, 256, 3))
	if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))
	if img_size != False:
       img = scipy.misc.imresize(img, img_size)
	return img

def init_dataset(dataset_path):


def set_batch_size(batch_size):

def get_next_batch():
