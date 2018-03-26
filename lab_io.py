import cv2
import numpy as np

def preprocess(img):
  # bgr to rgb
  img = img[...,::-1]
  # shape (h, w, d) to (1, h, w, d)
  img = img[np.newaxis,:,:,:]
  img -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  return img

def postprocess(img):
  img += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
  # shape (1, h, w, d) to (h, w, d)
  img = img[0]
  img = np.clip(img, 0, 255).astype('uint8')
  # rgb to bgr
  img = img[...,::-1]
  return img
  
def write_image(path, img):
  img = postprocess(img)
  cv2.imwrite(path, img)
  
def get_content_img(path):
    max_size = 512
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    h, w, d = img.shape
    mx = max_size
    # resize if > max size
    if h > w and h > mx:
        w = (float(mx) / float(h)) * w
        img = cv2.resize(img, dsize=(int(w), mx), interpolation=cv2.INTER_AREA)
    if w > mx:
        h = (float(mx) / float(w)) * h
        img = cv2.resize(img, dsize=(mx, int(h)), interpolation=cv2.INTER_AREA)
    
    img = preprocess(img)
    return img

def get_style_image(path, content_img):
    _, ch, cw, cd = content_img.shape
    # bgr image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
    img = preprocess(img)
    return img