import pandas as pd
from joblib import Parallel, delayed
import os
import numpy as np
import cv2
import random
import shutil
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.append("..")
from prepareSlices import *

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']



def create_slice_grid(patID):
    params = [
        (10, 15, "Small size, low visibility", 0, 0),
        (20, 15, "Large size, low visibility", 5, 0),
        (10, 33, "Small size, high visibility", 0, 1),
        (20, 33, "Large size, high visibility", 5, 1),
    ]

    fig = plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
    gs = gridspec.GridSpec(5, 10, figure=fig, hspace=0.1, wspace=0.15,
        height_ratios=[0.22, 1, 1, 1, 1],
        width_ratios=[0.25,1,1,1,1, 0.25,1,1,1,1])

    ax_label0 = fig.add_subplot(gs[0, 1:3])
    ax_label0.text(0.5, 0.5, 'Label 0', ha='center', va='center', fontsize=12)
    ax_label0.axis('off')

    ax_label1 = fig.add_subplot(gs[0, 3:5])
    ax_label1.text(0.5, 0.5, 'Label 1', ha='center', va='center', fontsize=12)
    ax_label1.axis('off')

    ax_label0 = fig.add_subplot(gs[0, 6:8])
    ax_label0.text(0.5, 0.5, 'Label 0', ha='center', va='center', fontsize=12)
    ax_label0.axis('off')

    ax_label1 = fig.add_subplot(gs[0, 8:])
    ax_label1.text(0.5, 0.5, 'Label 1', ha='center', va='center', fontsize=12)
    ax_label1.axis('off')

    ax_diff = fig.add_subplot(gs[1:3, 0])
    ax_diff.text(0.5, 0.5, 'Different total area', ha='center', va='center', fontsize=12, rotation=90)
    ax_diff.axis('off')

    ax_same = fig.add_subplot(gs[3:5, 0])
    ax_same.text(0.5, 0.5, 'Same total area', ha='center', va='center', fontsize=12, rotation=90)
    ax_same.axis('off')

    for i, (tot, bri, txt, cs, rs) in enumerate(params):
        for label in [0,1]:
            for rel in ['different', 'same']:
                ofs = 0 if rel == 'different' else 2
                col_start = 1+cs+2*label
                row_start = 1+rs+ofs
                img_path = f"./paper/Slice_{rel}_{tot}_{bri}_{label}.png"
                print (col_start, row_start, img_path)
                ax = fig.add_subplot(gs[row_start:row_start+1, col_start:col_start+2])
                img = Image.open(img_path)
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax.set_title(' ', fontsize=7, pad=0)
                ax.axis('off')
                if label == 0:
                    ax = fig.add_subplot(gs[row_start:row_start+1, col_start:col_start+4])
                    ax.text(0.5, 1.02, txt, ha='center', va='top', fontsize=8)
                    ax.axis('off')


    plt.savefig('../paper/Figure_2.png', dpi=700, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    try:
        shutil.rmtree("paper")
        shutil.rmtree("slices")
    except:
        pass
    os.makedirs('paper', exist_ok=True)

    patID = "Lipo-038"
    db = patID.split("-")[0]
    for d in [db]:
        data = getData(d, pinfoPath = "../data", radDBPath = "../data/radDB")
        data = data.sort_values(["Patient"])
        data = data.query("Patient == @patID").reset_index(drop = True).copy()
        for label in [0,1]:
            data.at[0,"Label"] = label
            for relativeSize in ['different', 'same']:
                for brightnessDiff in [15, 33]:
                    for totalSize in [20, 10]: # first the largest case so that coordinates fit for smaller case as well
                        data = addCoords(data, d, './slices', relativeSize, totalSize, brightnessDiff)
                        # and copy it away
                        img = cv2.imread(f"./slices/{d}/{patID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.png")
                        mask = cv2.imread(f"./slices/{d}/{patID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25_mask.png", 0)
                        coords = np.argwhere(mask > 0)
                        y0, x0 = coords.min(axis=0)
                        y1, x1 = coords.max(axis=0) + 1
                        cropped = img[y0:y1, x0:x1]
                        cv2.imwrite(f"./paper/Slice_{relativeSize}_{totalSize}_{brightnessDiff}_{label}.png", cropped)
    create_slice_grid(patID)

#
