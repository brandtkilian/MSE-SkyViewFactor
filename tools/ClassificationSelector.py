import numpy as np
import cv2
import os, re
from tools.FileManager import FileManager
import shutil


def beginSelection(src_folder, pred_folder, out_folder, skip=0):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    reg = r'\w+\.(jpg|jpeg|png)'
    preds = sorted([f for f in os.listdir(pred_folder) if re.match(reg, f.lower())])
    srcs = sorted([f for f in os.listdir(src_folder) if re.match(reg, f.lower())])

    assert len(srcs) == len(preds)
    length = len(srcs)

    i = skip if skip < length else 0

    while i < length:
        pred = FileManager.LoadImage(preds[i], pred_folder)
        src = FileManager.LoadImage(srcs[i], src_folder)

        pred = cv2.resize(pred, (480, 480), interpolation=cv2.INTER_NEAREST)
        src = cv2.resize(src, (480, 480), interpolation=cv2.INTER_CUBICyy)


        sky_mask, veg_mask, build_mask = cv2.split(pred)

        sky = cv2.bitwise_and(src, src, None, sky_mask)
        veg = cv2.bitwise_and(src, src, None, veg_mask)
        build = cv2.bitwise_and(src, src, None, build_mask)

        empty = np.zeros(src.shape, np.uint8)

        first_row = np.concatenate((src, pred, empty), axis=1)
        second_row = np.concatenate((sky, veg, build), axis=1)

        tot = np.concatenate((first_row, second_row), axis=0)

        cv2.imshow(preds[i], tot)

        k = cv2.waitKey(0) & 255

        pred_comp_path = os.path.join(pred_folder, preds[i])
        src_comp_path = os.path.join(src_folder, srcs[i])

        if k == 27:  # esc to exit
            break

        elif k == ord('y'): # yes keep
            shutil.copy(pred_comp_path, out_folder)
            shutil.copy(src_comp_path, out_folder)
            i += 1
        elif k == ord('n'): # no don't keep
            i += 1
            cv2.destroyAllWindows()
            continue
        cv2.destroyAllWindows()

    if i == length:
        print "The work is complete, bravo !"
    else:
        print "Next time you can start skipping %d images" % i
