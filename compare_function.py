# -*- conding:utf-8 -*-
import numpy as np
import tensorflow.keras.backend as K
import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, distance_transform_edt
from scipy.ndimage import binary_erosion
#import surface_distance as surfdist
import glob
import tqdm
from PIL import Image
import cv2
import os
from sklearn.metrics import f1_score
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import xlsxwriter
#tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

tf.disable_v2_behavior()

import sys
import os
import logging
import re
import functools
import fnmatch
import numpy as np
from scipy.ndimage.morphology import generate_binary_structure, distance_transform_edt
from scipy.ndimage import binary_erosion

import surface_distance as surfdist

def preprocessing_accuracy(label_true, label_pred, n_class=2):
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype='int8')
    label_true = np.asarray(label_true, dtype='int8')
    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 8)
    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)
    return label_true, label_pred
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result, reference = preprocessing_accuracy(reference, result)
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result):
        return 5000
        #raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        return 5000
        #raise RuntimeError('The second supplied array does not contain any binary object.')
        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    return sds


import numpy as np
from scipy.spatial.distance import cdist


#Dataset b
def hausdorff_distance2(prediction, ground_truth):
    # Convert prediction and ground_truth masks to 1D arrays of coordinates
    prediction_coords = np.array(np.where(prediction)).T
    ground_truth_coords = np.array(np.where(ground_truth)).T

    if len(prediction_coords) == 0 or len(ground_truth_coords) == 0:
        return None

    # Compute the distance matrix between all pairs of points in the two masks
    distances = cdist(prediction_coords, ground_truth_coords)

    # Calculate the Hausdorff distances in both directions
    hausdorff_forward = np.max(np.min(distances, axis=1))
    hausdorff_backward = np.max(np.min(distances, axis=0))

    # Return the maximum of the two Hausdorff distances
    return max(hausdorff_forward, hausdorff_backward)
from numba import jit
from scipy.spatial.distance import directed_hausdorff
#BUSI
@jit
def hausdorff_distance(prediction, ground_truth):
    # Convert prediction and ground_truth masks to 1D arrays of coordinates
    prediction_coords = np.array(np.where(prediction)).T
    ground_truth_coords = np.array(np.where(ground_truth)).T

    if len(prediction_coords) == 0 or len(ground_truth_coords) == 0:
        return None

    # Calculate the directed Hausdorff distance
    forward_hausdorff_distance = directed_hausdorff(prediction_coords, ground_truth_coords)[0]
    backward_hausdorff_distance = directed_hausdorff(ground_truth_coords, prediction_coords)[0]

    # Return the maximum of the two Hausdorff distances
    return max(forward_hausdorff_distance, backward_hausdorff_distance)

def ASSD(y_true, y_pred):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    surface_distances = surfdist.compute_surface_distances(
        y_true, y_pred, spacing_mm=(1.0, 1.0, 1.0))
    avg_surf_dist = surfdist.compute_average_surface_distance(
        surface_distances)
    #(gt2pre, pre2gt)
    pre2gt = avg_surf_dist[1]
    return pre2gt

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.
    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.
    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.
    See also
    --------
    :func:`hd`
    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    #
    hd95_mean = np.nanmean(hd95)
    return hd95_mean

def tf_repeat(tensor, repeats):
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor


def cal_base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN

def PA(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    ACC = (TP + TN) / (TP + FP + FN + TN + K.epsilon())
    return ACC

def IoU(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    iou = TP / (TP + FP + FN + K.epsilon())
    return iou

def Recall(y_true, y_pred):
    """ recall or sensitivity """
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SE = TP / (TP + FN + K.epsilon())
    return SE

def Precision(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    PC = TP / (TP + FP + K.epsilon())
    return PC

def Specificity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SP = TN / (TN + FP + K.epsilon())
    return SP

def Sensitivity(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    SEN = TP / (TP + FN + K.epsilon())
    return SEN

def F1_socre(y_true, y_pred):
    SE = Recall(y_true, y_pred)
    PC = Precision(y_true, y_pred)
    F1 = 2 * SE * PC / (SE + PC + K.epsilon())
    return F1

def Dice(y_true, y_pred):
    TP, TN, FP, FN = cal_base(y_true, y_pred)
    DC = 2*TP/(2*TP+FP+FN)
    return DC

def dice_coef(y_true, y_pred, smooth = 1):
    y_true_f = K.flatten(y_true) #/255.0
    y_pred_f = K.flatten(y_pred) #/255.0
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth = 1):
    y_true = K.flatten(y_true) #/255.0
    y_pred = K.flatten(y_pred) #/255.0
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true)+K.sum(y_pred)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

PAlist = []
MPAlist = []
Sensitivitylist = []
IoUlist = []
MIoUlist = []
Precisionlist = []
Recalllist = []
Specificitylist = []
F1_socrelist = []
Dicelist = []
HDlist = []
ASSDlist = []
Hingelist = []

# excle1 = xlsxwriter.Workbook("./BUSI3.xlsx")
# worksheet = excle1.add_worksheet()
# worksheet.write(0, 0, "image_name")
# # worksheet.write(0, 1, "PA")
# worksheet.write(0, 1, "IoU")
# works
# heet.write(0, 2, "Precision")
# worksheet.write(0, 3, "Recall")
# worksheet.write(0, 4, "Specificity")
# #worksheet.write(0, 6, "F1_socre")
# worksheet.write(0, 5, "Dice")
# # worksheet.write(0,8,"HD")
# # worksheet.write(0,9,"ASSD")

os.environ['CUDA_VISIBLE_DEVICES'] = '0' #GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' #CPU

image_path = './out/2/' # 72 81
# image_path = './config/B/Unet++/1/'# 23
# image_path = './config/DDTI/GLAN/3/'
# image_path = './config/BUSI/Total/GLAN/Sub/4_sub/'


# mask_path = "./BUSI/CV_all/3/test_annot/"3
# mask_path = "./BUSI/BUSI_1//test_annot/"
# mask_path = './DatasetB/CV_all/1/test_annot/'
#mask_path = './Thyroid Dataset/DDTI dataset/CV_all/4/test_annot/'
mask_path =  './Thyroid Dataset/DDTI dataset/CV_all2/2/test_annot/'
# mask_path = './combine/BIran/4/test_annot/'


filelist = os.listdir(mask_path)
i = 0

for item in filelist:
    print(item)
    i = i + 1
    print(i)
    image = cv2.imread(image_path + item, cv2.IMREAD_GRAYSCALE)/255.0
    mask = cv2.imread(mask_path + item, cv2.IMREAD_GRAYSCALE)/255.0
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    Himage = cv2.imread(image_path + item,cv2.IMREAD_GRAYSCALE)/255.0
    Hmask = cv2.imread(mask_path + item, cv2.IMREAD_GRAYSCALE)/255.0

    hd = hausdorff_distance(Himage, Hmask)


    # pa = PA(image, mask)
    dice = Dice(image, mask)
    iou = IoU(image, mask)
    precision = Precision(image, mask)
    recall = Recall(image, mask)
    specificity = Specificity(image, mask)
    sensitivity = Sensitivity(image,mask)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #PAlist.append(sess.run(pa))
        IoUlist.append(sess.run(iou))
        Precisionlist.append(sess.run(precision))
        Recalllist.append(sess.run(recall))
        Specificitylist.append(sess.run(specificity))
        Dicelist.append(sess.run(dice))
        Sensitivitylist.append(sess.run(sensitivity))
        #Hingelist.append(sess.run(hinge))
        HDlist.append(hd)
        #ASSDlist.append(assd)
        print(hd)
        # # #print(item)
        # # # print("PA:%f   IoU:%f  Precision:%f  Recall:%f  Specificity:%f   F1_socre:%f    Dice:%f"
        # # #       % (sess.run(pa), sess.run(iou), sess.run(precision), sess.run(recall), sess.run(specificity),
        # # #          sess.run(f1_score), sess.run(dice)))
        # #
        # worksheet.write(i, 0, item)
        # #worksheet.write(i, 1, sess.run(pa))
        # worksheet.write(i, 1, sess.run(iou))
        # worksheet.write(i, 2, sess.run(precision))
        # worksheet.write(i, 3, sess.run(recall))
        # worksheet.write(i, 4, sess.run(specificity))
        # #worksheet.write(i, 6, sess.run(f1_score))
        # worksheet.write(i, 5, sess.run(dice))
        # # worksheet.write(i,8,hd)
        # # worksheet.write(i,9,assd)


print(HDlist)

# new = []
# for index, value in enumerate(HDlist):
#
#     if value<300:
#         new.append(value)
#
# print()
# print(new)

#print("MPA:%f" % (sum(PAlist) / i))

valid_values = [value for value in HDlist if value is not None]
average_value = sum(valid_values) / len(valid_values)


# Format the average_value to print up to two decimal places (0.01)
formatted_average = "{:.6}".format(average_value)

print("MIoU:%f" % (sum(IoUlist) / i))
print("MPrecision:%f" % (sum(Precisionlist) / i))
print("MRecall:%f" % (sum(Recalllist) / i))
print("MSpecificity:%f" % (sum(Specificitylist) / i))
# print("ASSD:%f" % (sum(ASSDlist) / i))
#print("HD:%f" % (np.mean(HDlist)))
print("HD:", formatted_average)
print("Dice:%f" % (sum(Dicelist) / i))


#print("MHinge:%f" % (sum(Hingelist) / i))



# print("HD:%f" % (sum(HDlist) / i))
# print("ASSD:%f" % (sum(ASSDlist) / i))

# if i != 0:
#     #worksheet.write(i + 2, 1, sum(PAlist) / i)
#     worksheet.write(i + 2, 1, sum(IoUlist) / i)
#     worksheet.write(i + 2, 2, sum(Precisionlist) / i)
#     worksheet.write(i + 2, 3, sum(Recalllist) / i)
#     worksheet.write(i + 2, 4, sum(Specificitylist) / i)
#     #worksheet.write(i + 2, 6, sum(F1_socrelist) / i)
#     worksheet.write(i + 2, 5, sum(Dicelist) / i)
#     # worksheet.write(i+2,8,sum(HDlist) / i)
#     # worksheet.write(i+2,9,sum(ASSDlist) / i)
#     excle1.close()
#
#     # print("MPA:%f" % (sum(PAlist) / i))
#     # print("MIoU:%f" % (sum(IoUlist) / i))
#     # print("MPrecision:%f" % (sum(Precisionlist) / i))
#     # print("MRecall:%f" % (sum(Recalllist) / i))
#     # print("MSpecificity:%f" % (sum(Specificitylist) / i))
#     # print("MF1_score:%f" % (sum(F1_socrelist) / i))
#     # print("Dice:%f" % (sum(Dicelist) / i))
#     # print("HD:%f" % (sum(HDlist) / i))
#     # print("ASSD:%f" % (sum(ASSDlist) / i))
