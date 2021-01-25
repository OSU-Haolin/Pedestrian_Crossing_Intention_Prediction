import sys
import PIL
import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from sklearn.metrics import accuracy_score

# Data utilities
def flip_pose(pose):
    """
    Flips a given pose coordinates
    Args:
        pose: The original pose
    Return:
        Flipped poses
    """
    # [nose(0,1), neck(2,3), Rsho(4,5),   Relb(6,7),   Rwri(8,9),
    # 						 Lsho(10,11), Lelb(12,13), Lwri(14,15),
    #						 Rhip(16,17), Rkne(18,19), Rank(20,21),
    #                        Lhip(22,23), Lkne(24,25), Lank(26,27),
    #						 Leye(28,29), Reye (30,31),
    #						 Lear(32,33), Rear(34,35)]
    flip_map = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25,
                26, 27, 16, 17, 18, 19, 20, 21, 30, 31, 28, 29, 34, 35, 32, 33]
    new_pose = pose.copy()
    flip_pose = [0] * len(new_pose)
    for i in range(len(new_pose)):
        if i % 2 == 0 and new_pose[i] != 0:
            new_pose[i] = 1 - new_pose[i]
        flip_pose[flip_map[i]] = new_pose[i]
    return flip_pose


def get_pose(img_sequences,
             ped_ids, file_path,
             data_type='train',
             dataset='pie'):
    """
    Reads the pie poses from saved .pkl files
    Args:
        img_sequences: Sequences of image names
        ped_ids: Sequences of pedestrian ids
        file_path: Path to where poses are saved
        data_type: Whether it is for training or testing
    Return:
         Sequences of poses
    """

    print('\n#####################################')
    print('Getting poses %s' % data_type)
    print('#####################################')
    poses_all = []
    set_poses_list = [x for x in os.listdir(file_path) if x.endswith('.pkl')]
    set_poses = {}
    for s in set_poses_list:
        with open(os.path.join(file_path, s), 'rb') as fid:
            try:
                p = pickle.load(fid)
            except:
                p = pickle.load(fid, encoding='bytes')
        set_poses[s.split('.pkl')[0].split('_')[-1]] = p
    i = -1
    for seq, pid in zip(img_sequences, ped_ids):
        i += 1
        update_progress(i / len(img_sequences))
        pose = []
        for imp, p in zip(seq, pid):
            flip_image = False
            
            if dataset == 'pie':
                set_id = imp.split('/')[-3]
            elif dataset == 'jaad':
                set_id = 'set01'
            
            vid_id = imp.split('/')[-2]
            img_name = imp.split('/')[-1].split('.')[0]
            if 'flip' in img_name:
                img_name = img_name.replace('_flip', '')
                flip_image = True
            k = img_name + '_' + p[0]
            if k in set_poses[set_id][vid_id].keys():
                # [nose, neck, Rsho, Relb, Rwri, Lsho, Lelb, Lwri, Rhip, Rkne,
                #  Rank, Lhip, Lkne, Lank, Leye, Reye, Lear, Rear, pt19]
                if flip_image:
                    pose.append(flip_pose(set_poses[set_id][vid_id][k]))
                else:
                    pose.append(set_poses[set_id][vid_id][k])
            else:
                pose.append([0] * 36)
        poses_all.append(pose)
    poses_all = np.array(poses_all)
    return poses_all


def jitter_bbox(img_path, bbox, mode, ratio):
    """
    Jitters the position or dimensions of the bounding box.
    Args:
        img_path: The to the image
        bbox: The bounding box to be jittered
        mode: The mode of jitterring. Options are,
          'same' returns the bounding box unchanged
          'enlarge' increases the size of bounding box based on the given ratio.
          'random_enlarge' increases the size of bounding box by randomly sampling a value in [0,ratio)
          'move' moves the center of the bounding box in each direction based on the given ratio
          'random_move' moves the center of the bounding box in each direction by randomly
                        sampling a value in [-ratio,ratio)
        ratio: The ratio of change relative to the size of the bounding box.
           For modes 'enlarge' and 'random_enlarge'
           the absolute value is considered.
    Return:
        Jitterred bounding boxes
    """

    assert (mode in ['same', 'enlarge', 'move', 'random_enlarge', 'random_move']), \
        'mode %s is invalid.' % mode

    if mode == 'same':
        return bbox

    img = load_img(img_path)

    if mode in ['random_enlarge', 'enlarge']:
        jitter_ratio = abs(ratio)
    else:
        jitter_ratio = ratio

    if mode == 'random_enlarge':
        jitter_ratio = np.random.random_sample() * jitter_ratio
    elif mode == 'random_move':
        # for ratio between (-jitter_ratio, jitter_ratio)
        # for sampling the formula is [a,b), b > a,
        # random_sample * (b-a) + a
        jitter_ratio = np.random.random_sample() * jitter_ratio * 2 - jitter_ratio

    jit_boxes = []
    for b in bbox:
        bbox_width = b[2] - b[0]
        bbox_height = b[3] - b[1]

        width_change = bbox_width * jitter_ratio
        height_change = bbox_height * jitter_ratio

        if width_change < height_change:
            height_change = width_change
        else:
            width_change = height_change

        if mode in ['enlarge', 'random_enlarge']:
            b[0] = b[0] - width_change // 2
            b[1] = b[1] - height_change // 2
        else:
            b[0] = b[0] + width_change // 2
            b[1] = b[1] + height_change // 2

        b[2] = b[2] + width_change // 2
        b[3] = b[3] + height_change // 2

        # Checks to make sure the bbox is not exiting the image boundaries
        b = bbox_sanity_check(img.size, b)
        jit_boxes.append(b)
    # elif crop_opts['mode'] == 'border_only':
    return jit_boxes


def squarify(bbox, squarify_ratio, img_width):
    """
    Changes the dimensions of a bounding box to a fixed ratio
    Args:
        bbox: Bounding box
        squarify_ratio: Ratio to be changed to
        img_width: Image width
    Return:
        Squarified boduning boxes
    """
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change / 2
    bbox[2] = bbox[2] + width_change / 2
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0

    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        # bbox[1] = str(-float(bbox[3]) + img_dimensions[0])
        bbox[0] = bbox[0] - bbox[2] + img_width
        bbox[2] = img_width
    return bbox


def update_progress(progress):
    """
    Shows the progress
    Args:
        progress: Progress thus far
    """
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(barLength * progress))
    text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def img_pad_pil(img, mode='warp', size=224):
    """
    Pads and/or resizes a given image
    Args:
        img: The image to be coropped and/or padded
        mode: The type of padding or resizing. Options are,
            warp: crops the bounding box and resize to the output size
            same: only crops the image
            pad_same: maintains the original size of the cropped box  and pads with zeros
            pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
                        the desired output size in that direction while maintaining the aspect ratio. The rest
                        of the image is	padded with zeros
            pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
                    in which case it scales the image down, and then pads it
        size: Target size of image
    Return:
        Padded image
    """
    assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size, size), PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size) / max(img_size)
        if mode == 'pad_resize' or \
                (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size - img_size[0]) // 2,
                                   (size - img_size[1]) // 2))
        return padded_image

def img_pad(img, mode='warp', size=224):
    """
    Pads and/or resizes a given image
    Args:
        img: The image to be coropped and/or padded
        mode: The type of padding or resizing. Options are,
            warp: crops the bounding box and resize to the output size
            same: only crops the image
            pad_same: maintains the original size of the cropped box  and pads with zeros
            pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
                        the desired output size in that direction while maintaining the aspect ratio. The rest
                        of the image is	padded with zeros
            pad_fit: maintains the original size of the cropped box unless the image is bigger than the size
                    in which case it scales the image down, and then pads it
        size: Target size of image
    Return:
        Padded image
    """
    assert (mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = np.copy(img)
    if mode == 'warp':
        warped_image = cv2.resize(img, (size, size))
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same', 'pad_resize', 'pad_fit']:
        img_size = image.shape[:2][::-1] # original size is in (height, width)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or \
                (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0] * ratio), int(img_size[1] * ratio)])
            image = cv2.resize(image, img_size)
        padded_image = np.zeros((size, size)+(image.shape[-1],), dtype=img.dtype)
        w_off = (size-img_size[0])//2
        h_off = (size-img_size[1])//2
        padded_image[h_off:h_off + img_size[1], w_off:w_off+ img_size[0],:] = image
        return padded_image


def bbox_sanity_check(img_size, bbox):
    """
    Checks whether  bounding boxes are within image boundaries.
    If this is not the case, modifications are applied.
    Args:
        img_size: The size of the image
        bbox: The bounding box coordinates
    Return:
        The modified/original bbox
    """
    img_width, img_heigth = img_size
    if bbox[0] < 0:
        bbox[0] = 0.0
    if bbox[1] < 0:
        bbox[1] = 0.0
    if bbox[2] >= img_width:
        bbox[2] = img_width - 1
    if bbox[3] >= img_heigth:
        bbox[3] = img_heigth - 1
    return bbox


def get_path(file_name='',
             sub_folder='',
             save_folder='models',
             dataset='pie',
             save_root_folder='data/'):
    """
    Generates paths for saving model and config data.
    Args:
        file_name: The actual save file name , e.g. 'model.h5'
        sub_folder: If another folder to be created within the root folder
        save_folder: The name of folder containing the saved files
        dataset: The name of the dataset used
        save_root_folder: The root folder
    Return:
        The full path and the path to save folder
    """
    save_path = os.path.join(save_root_folder, dataset, save_folder, sub_folder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return os.path.join(save_path, file_name), save_path


# Optical flow utilities
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def read_flow_file(optflow_path):
    with open(optflow_path, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        data2d = None
        assert tag == 202021.25, 'Incorrect .flo file, {}'.format(optflow_path)
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        return np.resize(data2d, (h, w, 2))
def write_flow(flow, optflow_path):
    with open(optflow_path, 'wb') as f:
        magic = np.array([202021.25], dtype=np.float32)
        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        magic.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel
def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def tte_weighted_acc(tte, gt, y, weights='quadratic'):
    """
    A function to compute time-to-event (TTE) weighted accuracy: 
    1) computes accuracy for unique TTEs in the list,
    2) computes weighted average of accuracy scores assigning higher weight to higher TTEs.
    
    Args:
        tte: array of TTE values for each sample
        gt: ground truth sample class
        y: predicted sample class
        weights: linear or quadratic
    """

    sort_idx = np.argsort(tte)
    tte_sorted = tte[sort_idx]
    unq_tte_first = np.concatenate(([True], tte_sorted[1:] != tte_sorted[:-1]))
    unq_tte = tte_sorted[unq_tte_first]
    unq_tte_count = np.diff(np.nonzero(unq_tte_first)[0])
    unq_tte_index = np.split(sort_idx, np.cumsum(unq_tte_count))

    acc_tte = []
    for tte, tte_idx in zip(unq_tte, unq_tte_index):
        acc_tte.append(accuracy_score(gt[tte_idx], np.round(y[tte_idx])))

    assert weights in ['linear', 'quadratic'], 'Weights type {} is not implemented!'.format(weights)

    if weights == 'quadratic':
        unq_tte = np.square(unq_tte)

    acc_tte = np.sum(np.multiply(acc_tte, unq_tte)/np.sum(unq_tte))

    return acc_tte