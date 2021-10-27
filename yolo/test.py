import cv2
import numpy as np
def load_image(img_path: str):
    """
    Loads an encoded image as an array of bytes.

    This is a typical approach you'd like to use in DALI backend.
    DALI performs image decoding, therefore this way the processing
    can be fully offloaded to the GPU.
    """
    # f = open(img_path, 'rb')
    # img_data = np.frombuffer(f.read(), dtype='uint8')
    # return img_data
    # return np.fromfile(img_path, dtype='a4')
    return np.fromfile(img_path, dtype='uint8')

def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
    print(image_padded,gt_boxes)
    print(image_padded.shape,gt_boxes.shape)
    print(type(image_padded,gt_boxes))
    return image_padded, gt_boxes


path_images="person_dog.jpg"
# Preprocess the images into input data according to model
# requirements
image_data = []

input_size = 416

original_image = cv2.imread("person_dog.jpg")
print(original_image.shape)
print(type(original_image))
original_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
print(original_image1.shape)
print(type(original_image1))
original_image_size = original_image1.shape[:2]
print(original_image_size)
image_data_1 = image_preprocess(np.copy(original_image1), [input_size, input_size]).astype(dtype=np.uint8)
image_data_2 = np.expand_dims(image_data_1,axis=0)
# con_uint8=load_image(image_data_1)
image_data.append(image_data_1)
x=np.array(image_data_1)
print(image_data_2.shape)
print(image_data_2.dtype)
# image=load_image("person_dog.jpg")
# image=np.expand_dims(image,axis=0)
# print(image.shape)
# print(image.dtype)
# print(type(con_uint8))
# print(con_uint8.shape)




