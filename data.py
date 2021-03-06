import os
import scipy
import numpy as np

_content_targets = {}
_batch_shape = 0
_iteration = 0
_batch_size = 0
_data_num = 0

def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img

def img_fit_to(img, max_size=512):
     # bgr image
    h, w, d = img.shape
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = scipy.misc.imresize(img, (mx, int(w), d))
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = scipy.misc.imresize(img, (int(h), mx, d))

    return img

def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)

def init_dataset(dataset_path, batch_shape):
    _content_targets = _get_files(dataset_path)
    _data_num = len(_content_targets)
    _batch_shape = batch_shape
    _batch_size = _batch_shape[0]

def get_next_batch():
    if _iteration + _batch_size > _data_num:
	    return np.array()

    x_batch = np.zeros(_batch_shape, dtype=np.float32)
    for j, img_p in enumerate(_content_targets[_iteration:(_iteration + _batch_size)]):
        x_batch[j] = get_img(img_p, batch_shape[1:4]).astype(np.float32)

    _iterations += _batch_size

    return x_batch


def _list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        files.extend(filenames)
        break

    return files
    
def _get_files(img_dir):
    files = _list_files(img_dir)
    return [os.path.join(img_dir,x) for x in files]
