import pandas as pd
from joblib import Parallel, delayed
import os
import nibabel as nib
import nibabel.processing as nibp
import numpy as np
import cv2
import random
import shutil
import scipy.ndimage as ndimage
from pattern import *



def getData(dataset, pinfoPath = "./data/", radDBPath = './data/radDB'):
    data = pd.read_csv(f"{pinfoPath}/pinfo_{dataset}.csv")
    for i, (idx, row) in enumerate(data.iterrows()):
        image = os.path.join(radDBPath, row["Patient"], "image.nii.gz")
        if dataset == "CRLM":
            mask = os.path.join(radDBPath, row["Patient"], "segmentation_lesion0_RAD.nii.gz")
        else:
            mask = os.path.join(radDBPath, row["Patient"], "segmentation.nii.gz")
        data.at[idx, "Image"] = image
        data.at[idx, "mask"] = mask
    return data



def find_multiple_disc_positions(mask, num_positions=4, d_size=64, dbg = False):
    valid_coords = []
    step_size = 2
    for y in range(0, mask.shape[0] - d_size + 1, step_size):
        for x in range(0, mask.shape[1] - d_size + 1, step_size):
            if np.sum(mask[y:y+d_size, x:x+d_size]) / (d_size**2) > 0.96: # roughly 2 pixels off?
                valid_coords.append((y, x))
    if dbg == True:
        print ("Testing ", len(valid_coords), "coords now")

    if len(valid_coords) < num_positions:
        raise Exception ("Not enough space")

    random.shuffle(valid_coords)
    selected = []
    for coord in valid_coords:
        if len(selected) >= num_positions: break
        if all(abs(coord[0] - s[0]) > d_size//2 or abs(coord[1] - s[1]) > d_size//2 for s in selected):
            selected.append(coord)

    if len(selected) != num_positions:
        return None

    return selected



def process_single_coord(row, dataID, slicesPath, relativeSize, totalSize = 10, brightnessDiff = 10):
    f, fmask, label = row["Image"], row["mask"], row["Label"]
    img, seg = nib.load(f), nib.load(fmask)
    seg_data = seg.get_fdata()

    slice_sums = np.sum(seg_data, axis=(0, 1))
    max_idx = np.argmax(slice_sums)

    os.makedirs(f'{slicesPath}/{dataID}', exist_ok=True)

    img_slice = img.get_fdata()[:, :, max_idx]
    seg_slice = seg_data[:, :, max_idx]

    zoomfct = 0.25  # in mm

    zooms = img.header.get_zooms()[:2]
    scale_factors = [z / zoomfct for z in zooms]

    img_slice = ndimage.zoom(img_slice, scale_factors, order=3)
    seg_slice = ndimage.zoom(seg_slice, scale_factors, order=0)

    masked_pixels = img_slice[seg_slice > 0]
    min_val = masked_pixels.min()
    max_val = masked_pixels.max()
    subtract_val = (max_val - min_val) * brightnessDiff/100 # so 10=10/100=10%, 25=25/100=1/4 of range

    # if we fix  say 32 lesion size, it might be too small or large
    # y_coords, x_coords = np.where(seg_slice > 0)
    # y_min, y_max = y_coords.min(), y_coords.max()
    # x_min, x_max = x_coords.min(), x_coords.max()
    #
    # if x_max-x_min < y_max-y_min:
    #     d_size = (x_max-x_min)//16
    # else:
    #     d_size = (y_max-y_min)//16

    total_mask_pixels = masked_pixels.size
    d_size = int(np.sqrt(total_mask_pixels) * totalSize/100)

    min_physical_size = 1.0
    min_pixels = int(np.ceil(min_physical_size / zoomfct))
    # print (d_size)
    if d_size < min_pixels:
        print ("ERROR MIN")
        d_size = min_pixels

    # we always find first coordinates for larger disks
    if "Coords" in row and isinstance(row["Coords"], list):
        allpositions = row["Coords"]
    else:
        # technically, label is now 'do you see more than one disc?', so if label=1, yes, there are more discs.
        # we find here for one disc a larger area, so we can fit same and larger disc at the same position
        allpositions = None
        dbg = False
        while not allpositions and d_size >= min_pixels:
            try:
                allpositions = find_multiple_disc_positions(seg_slice, num_positions=4, d_size=2*d_size, dbg = dbg)
            except:
                pass
            if allpositions is None:
                d_size = d_size - 2
                dbg = True
                print ("Trying again with d_size", d_size)

    if not allpositions:
        print(row)
        raise Exception("Cannot find disc positions")

    # now we use what we need
    if label == 0:
        if relativeSize == 'same':
            d_size = d_size * 2
        positions = allpositions[0:1].copy() # first one only
    else:
        positions = allpositions.copy()

    print (label, d_size, "MIN", min_pixels, positions, row["Patient"])
    for pos in positions:
        patbmap = create_pattern("simple", d_size, frown=(label == 1))
        y, x = pos
        if y + d_size <= img_slice.shape[0] and x + d_size <= img_slice.shape[1]:
            pat_region = img_slice[y:y+patbmap.shape[0], x:x+patbmap.shape[1]].copy()
            darkening_mask = patbmap > 0
            pat_region[darkening_mask] = np.maximum(pat_region[darkening_mask] - subtract_val, min_val)
            img_slice[y:y+patbmap.shape[0], x:x+patbmap.shape[1]] = pat_region

    new_affine = np.eye(4)
    new_affine[0,0], new_affine[1,1] = 0.5, 0.5
    new_affine[2,2] = img.header.get_zooms()[2]

    nib.save(nib.Nifti1Image(img_slice, new_affine), f'{slicesPath}/{dataID}/{row["Patient"]}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.nii.gz')
    nib.save(nib.Nifti1Image(seg_slice, new_affine), f'{slicesPath}/{dataID}/{row["Patient"]}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25_mask.nii.gz')

    # can also create deep version now
    seg_slice = ((seg_slice > 0)*255).astype(np.uint8)

    # normalize mask
    masked_pixels = img_slice[seg_slice > 0]
    min_val = masked_pixels.min()
    max_val = masked_pixels.max()
    img_slice = np.zeros_like(img_slice)
    img_slice[seg_slice > 0] = (masked_pixels - min_val) / (max_val - min_val) * 255

    # 'zoom' in
    y_coords, x_coords = np.where(seg_slice > 0)
    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    img_slice = img_slice[y_min:y_max+1, x_min:x_max+1]
    seg_slice = seg_slice[y_min:y_max+1, x_min:x_max+1]
    img_slice = img_slice.astype(np.uint8)

    cv2.imwrite(f'{slicesPath}/{dataID}/{row["Patient"]}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.png', img_slice)
    cv2.imwrite(f'{slicesPath}/{dataID}/{row["Patient"]}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25_mask.png', seg_slice)

    return allpositions



def addCoords(data, dataID, slicesPath, relativeSize, totalSize, brightnessDiff):
    results = Parallel(n_jobs=16)(delayed(process_single_coord)(row, dataID, slicesPath, relativeSize, totalSize, brightnessDiff) for _, row in data.iterrows())
    data["Coords"] = results
    return data



def addRandomLabel(data):
    for (idx, row) in data.iterrows():
        data.at[idx, "Label"] = np.random.randint(0, 2)
    return data.copy()



if __name__ == '__main__':
    slicesPath = "./slices"
    try:
        shutil.rmtree(slicesPath)
    except:
        pass
    os.makedirs(f'{slicesPath}', exist_ok=True)

    for d in ["Desmoid", "CRLM", "GIST", "Lipo"]:
        data = getData(d)
        data = addRandomLabel(data)  # at 'same' pos same will work as well
        for relativeSize in ['different', 'same']:
            for brightnessDiff in [15, 33]:
                for totalSize in [20, 10]: # first the largest case so that coordinates fit for smaller case as well
                    data = addCoords(data, d, slicesPath, relativeSize, totalSize, brightnessDiff)
                    data.to_csv(f"./{slicesPath}/{d}_{relativeSize}_{totalSize}_{brightnessDiff}_coords_labels.csv", index=False)
                    df_export = data[['Patient', 'Label']].copy()
                    df_export = df_export.rename(columns={'Patient': 'PatientID', 'Label': 'Target'})
                    df_export['Target'] = df_export['Target'].astype(int)
                    df_export.to_csv(f"./{slicesPath}/{d}_{relativeSize}_{totalSize}_{brightnessDiff}_labels.csv", index=False)

#
