import numpy as np
import cv2
import pandas as pd
import os
from scipy import ndimage
import skimage.draw
import shutil
from skimage.morphology import binary_closing, binary_opening, disk
from skimage.measure import label
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
from skimage.draw import line
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
from radiomics import featureextractor
from sklearn.preprocessing import StandardScaler

from pattern import *


def extract_radiomic_features(dataID, relativeSize, totalSize, brightnessDiff):
    if os.path.exists(f"./slices/{dataID}_{relativeSize}_{totalSize}_{brightnessDiff}_features.csv") == True:
        return None

    # Extract radiomics features first
    if dataID == "CRLM" or dataID == "GIST":
        params = os.path.join("config/CT.yaml")
    else:
        params = os.path.join("config/MR.yaml")
    eParams = {"binWidth": 25, "force2D": True}
    extractor = featureextractor.RadiomicsFeatureExtractor(params, **eParams)
    extractor.enableImageTypeByName("LBP2D")

    labels_df = pd.read_csv(f"slices/{dataID}_{relativeSize}_{totalSize}_{brightnessDiff}_labels.csv")

    features_list = []
    for idx, row in labels_df.iterrows():
        ID = row['PatientID']
        print (row)
        label = row['Target']
        image_path = f"./slices/{dataID}/{ID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.nii.gz"
        mask_path =  f"./slices/{dataID}/{ID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25_mask.nii.gz"

        features = extractor.execute(image_path, mask_path)
        feature_dict = {
            'PatientID': ID,
            'Target': label
        }

        for key, value in features.items():
            if not key.startswith('diagnostics_'):
                feature_dict[key] = float(value)
        features_list.append(feature_dict)

    # normalize
    features_df = pd.DataFrame(features_list)
    include_cols = [col for col in features_df.columns if col not in ["Target", "PatientID"]]
    scaler = StandardScaler()
    features_df[include_cols] = scaler.fit_transform(features_df[include_cols])
    features_df.to_csv(f"./slices/{dataID}_{relativeSize}_{totalSize}_{brightnessDiff}_features.csv", index=False)
    return features_df


if __name__ == "__main__":
    for d in ["CRLM", "Desmoid", "GIST", "Lipo"]:
        for relativeSize in ['different', 'same']:
            for brightnessDiff in [15, 33]:
                for totalSize in [20, 10]: # first the largest case so that coordinates fit for smaller case as well
                    features = extract_radiomic_features(d, relativeSize, totalSize, brightnessDiff)

#
