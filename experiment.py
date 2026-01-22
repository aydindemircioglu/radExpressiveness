from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold

import shutil
import os
import sys
import pickle
import glob

from ultralytics import YOLO
import torch

from featureselection import *

import hashlib
from functools import partial
from joblib import dump, Parallel, delayed, load
import numpy as np
import pandas as pd
import random
import time

import optuna
import warnings
from optuna.exceptions import ExperimentalWarning
optuna.logging.set_verbosity(optuna.logging.FATAL)


n_outer_cv = 5
n_inner_cv = 5
num_repeats = 3


search_space = {
    'fs_method': ["LASSO", "ET"],
    'N': [2**k for k in range(0,6)], # N = 1 yields an error because of LASSO and SelectFromModel.
    'clf_method': ["RBFSVM", "RandomForest", "LogisticRegression", "NaiveBayes"],
    'RF_n_estimators': [25,50,100,250,500],
    'C_LR': [2**k for k in range(-5,5,2)],
    'C_SVM': [2**k for k in range(-5,5,2)]
}



def set_all_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmrf (fpath):
    try:
        shutil.rmtree(fpath)
    except:
        pass


def extract_features_from_yolo(model, image_path, device=0):
    features = []
    def hook(module, input, output):
        features.append(output.flatten().detach().cpu())
    handle = model.model.model[-2].register_forward_hook(hook)
    model.predict([image_path], imgsz=224, device=device, verbose=False)
    handle.remove()
    return features[0].numpy()



def extract_yolo_features (train_data, test_data, dataset, relativeSize, brightnessDiff, totalSize):
    rid = random.randint(10**11, 10**12 - 1)
    rmrf(f"./cache/{rid}")
    os.makedirs(f"./cache/{rid}")

    feature_accumulator = {}
    patient_targets = {}
    inner_cv = StratifiedKFold(n_splits=n_inner_cv)
    for i, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_data, train_data["Target"])):
        X_inner_train = train_data.iloc[inner_train_idx]
        X_inner_val = train_data.iloc[inner_val_idx]

        copyData(dataset, f"./cache/{rid}/", relativeSize, brightnessDiff, totalSize, [("train", X_inner_train), ("val", X_inner_val)])

        model = YOLO("yolo11n-cls.pt")
        run_name = f"fold_{i}"
        model.train(data=f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}", imgsz=224, erasing=0.0, epochs = 300, patience = 100,
            device=0, project=f'./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/runs', name = run_name, exist_ok = True)

        best_model_path = f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/runs/{run_name}/weights/best.pt"
        best_model = YOLO(best_model_path)

        val_probs, val_labels = get_probs_and_labels(best_model, f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}", "val")
        print(f"Fold {i+1} AUC: {roc_auc_score(val_labels, val_probs)}")

        with torch.no_grad():
            for dset in [train_data, test_data]:
                for pID in dset["PatientID"].values:
                    if pID not in patient_targets:
                        patient_targets[pID] = dset.query("PatientID == @pID").iloc[0]["Target"]
                    src_path = f"slices/{dataset}/{pID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.png"
                    feature_vector = np.array(extract_features_from_yolo(best_model, src_path))
                    if pID not in feature_accumulator:
                        feature_accumulator[pID] = feature_vector
                    else:
                        feature_accumulator[pID] += feature_vector
        # we have the code to do the 'full' training with taking mean across all trained model,
        # but what for, we know by the smiley and plus/minus experiment that the network performs perfectly,
        # even when being 'made worse' by this decision
        # note the reconstruct_dataframe function has now no divsion by n_inner_cv!
        break

    def reconstruct_dataframe(original_df):
        rows = []
        for pID in original_df["PatientID"].values:
            avg_features = feature_accumulator[pID] # / n_inner_cv
            tgt = patient_targets[pID]
            row = {"PatientID": pID, "Target": tgt}
            for idx, val in enumerate(avg_features):
                row[f"feat_{idx}"] = val
            rows.append(row)
        return pd.DataFrame(rows)

    # Build final DataFrames
    df_train_features = reconstruct_dataframe(train_data)
    df_test_features = reconstruct_dataframe(test_data)

    rmrf(f"./cache/{rid}")
    return df_train_features, df_test_features



def extract_rad_features(train_data, test_data, dataset, relativeSize, brightnessDiff, totalSize):
    fv = pd.read_csv(f"./slices/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}_features.csv").drop(["Target"], axis = 1)
    df_train_features = train_data.merge(fv, on="PatientID", how="left")
    df_test_features = test_data.merge(fv, on="PatientID", how="left")
    return df_train_features, df_test_features



def extract_features (df_all, train_ids, test_ids, dataset, method, relativeSize, brightnessDiff, totalSize):
    train_data = df_all[df_all["PatientID"].isin(train_ids)].reset_index(drop = True).copy()
    test_data = df_all[df_all["PatientID"].isin(test_ids)].reset_index(drop = True).copy()

    if method == "DeepRadiomics":
        df_train_features, df_test_features = extract_yolo_features(train_data, test_data, dataset, relativeSize, brightnessDiff, totalSize)
    elif method == "Radiomics":
        df_train_features, df_test_features = extract_rad_features(train_data, test_data, dataset, relativeSize, brightnessDiff, totalSize)
    else:
        raise Exception ("Unknown method")
    return df_train_features, df_test_features



def select_features(X, y, fs_method, N, feature_names=None):
    if fs_method == "LASSO":
        clf_fs = LogisticRegression(penalty='l1', max_iter=100, solver='liblinear', C=1, random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)
    elif fs_method == "MRMRe":
        mrmre_score_fct = partial(mrmre_score, nFeatures = N)
        fsel = SelectKBest(mrmre_score_fct, k = N)
    elif fs_method == "ET":
        clf_fs = ExtraTreesClassifier(random_state=42)
        fsel = SelectFromModel(clf_fs, prefit=False, max_features=N, threshold=-np.inf)
    X_selected = fsel.fit_transform(X, y)

    if feature_names is None:
        selected_names = None
    else:
        selected_mask = fsel.get_support()
        selected_names = [feature_names[i] for i in range(len(selected_mask)) if selected_mask[i]] if feature_names is not None else None

    return X_selected, fsel, selected_names



def getClassifier(best_params):
    if best_params['clf_method'] == "LogisticRegression":
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=best_params['C_LR'])
    elif best_params['clf_method'] == "NaiveBayes":
        clf = GaussianNB()
    elif best_params['clf_method'] == "RandomForest":
        clf = RandomForestClassifier(n_estimators=best_params['RF_n_estimators'])
    elif best_params['clf_method'] == "RBFSVM":
        clf = SVC(kernel="rbf", C=best_params['C_SVM'], gamma='auto', probability=True)
    return clf



# will apply inner CV
def inner_objective(trial, dataset, df_train, repeat, num_inner_splits=5):
    N = trial.suggest_categorical("N", search_space['N'])
    fs_method = trial.suggest_categorical("fs_method", search_space['fs_method'])

    clf_method = trial.suggest_categorical("clf_method", search_space['clf_method'])
    if clf_method == "LogisticRegression":
        C_LR = trial.suggest_categorical("C_LR", search_space['C_LR'])
        clf = LogisticRegression(max_iter=500, solver='liblinear', C=C_LR)
    elif clf_method == "NaiveBayes":
        clf = GaussianNB()
    elif clf_method == "RandomForest":
        RF_n_estimators = trial.suggest_categorical("RF_n_estimators", search_space['RF_n_estimators'])
        clf = RandomForestClassifier(n_estimators=RF_n_estimators)
    elif clf_method == "RBFSVM":
        C_SVM = trial.suggest_categorical("C_SVM", search_space['C_SVM'])
        clf = SVC(kernel="rbf", C=C_SVM, gamma='auto', probability=True)

    inner_cv = StratifiedKFold(n_splits=num_inner_splits)#, random_state=42)
    y_probs = []
    y_gt = []

    # prepare data
    for inner_train_idx, inner_val_idx in inner_cv.split(df_train, df_train["Target"]):
        y_inner_train = df_train.iloc[inner_train_idx]["Target"].values
        X_inner_train = df_train.iloc[inner_train_idx].drop(columns=["Target", "PatientID"]).values
        X_inner_train_selected, fsel, _ = select_features(X_inner_train, y_inner_train, fs_method, N)

        val_df = df_train.iloc[inner_val_idx]
        y_inner_val = val_df["Target"].values
        X_inner_val = val_df.drop(columns=["Target", "PatientID"]).values
        X_inner_val_selected = fsel.transform(X_inner_val)

        clf.fit(X_inner_train_selected, y_inner_train)
        y_prob = clf.predict_proba(X_inner_val_selected)[:, 1]

        if np.any(np.isnan(y_prob)):
            is_constant = np.all(X_inner_train_selected == X_inner_train_selected[0])
            if is_constant:
                # this can happen now, if the data is too small, happens with
                # Bhattacharyya. in that case we replace the probs randomly
                random_probs = np.random.random(size=len(y_prob))
                y_prob = random_probs / np.sum(random_probs)
            else:
                # this should never happen
                raise Exception("NaN values detected in non-constant features")

        y_probs.append(y_prob)
        y_gt.append(y_inner_val)

    y_prob_flat = [p for y in y_probs for p in y]
    y_true_flat = [gt for y in y_gt for gt in y]
    cv_auc = roc_auc_score(y_true_flat, y_prob_flat)
    trial.set_user_attr("auc_int", cv_auc)
    return cv_auc



def get_probs_and_labels(best_model, base_path, dset_name):
    probs = []
    labels = []
    for label in ["0","1"]:
        folder = f"{base_path}/{dset_name}/{label}/"
        files = os.listdir(folder)
        paths = [os.path.join(folder, f) for f in files]
        results = best_model.predict(paths, imgsz=224, device=0, verbose=False)
        for r in results:
            probs.append(r.probs.data[1].item())
            labels.append(int(label))
    return np.array(probs), np.array(labels)



def copyData(dataset, tgt_base, relativeSize, brightnessDiff, totalSize, dsets):
    for dset in dsets:
        for j in range(len(dset[1])):
            row = dset[1].iloc[j]
            pID = row["PatientID"]
            Target = row["Target"]
            os.makedirs(f'{tgt_base}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/{dset[0]}/{Target}', exist_ok=True)
            src_path = f"slices/{dataset}/{pID}_{relativeSize}_{totalSize}_{brightnessDiff}_0.25.png"
            tgt_path = f"{tgt_base}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/{dset[0]}/{Target}/{pID}.png"
            shutil.copyfile(src_path, tgt_path)


def evaluate(dataset, method, df_all, train_ids, test_ids, relativeSize, brightnessDiff, totalSize, repeat):
    if method == "Deep":
        # train yolo on train_ids and evaluate on test_ids
        inner_cv = StratifiedKFold(n_splits=n_inner_cv)
        auc_cv_int = []

        # prepare data, we only use one split
        rid = random.randint(10**11, 10**12 - 1)
        train_data = df_all[df_all["PatientID"].isin(train_ids)].reset_index(drop = True).copy()
        for inner_train_idx, inner_val_idx in inner_cv.split(train_data, train_data["Target"]):
            X_inner_train = train_data.iloc[inner_train_idx]
            X_inner_val = train_data.iloc[inner_val_idx]

            #prepare_YOLO_Data
            rmrf(f"./cache/{rid}")
            os.makedirs(f"./cache/{rid}")
            copyData(dataset, f"./cache/{rid}/", relativeSize, brightnessDiff, totalSize, [("train", X_inner_train), ("val", X_inner_val)])

            model = YOLO("yolo11n-cls.pt")
            model.train(data=f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/", imgsz=224, erasing=0.0, epochs = 300, patience = 100,
                device=0, project=f'./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/runs')
            best_model = YOLO(f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}/runs/train/weights/best.pt")

            val_probs, val_labels = get_probs_and_labels(best_model, f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}", "val")
            auc_cv_int.append(roc_auc_score(val_labels, val_probs))
            # here we take only the first model, just as before.
            break

        auc_cv_int = np.mean(auc_cv_int)
        print ("Deep, int AUC:", auc_cv_int)

        # Evaluate on test
        X_test_data = df_all[df_all["PatientID"].isin(test_ids)].reset_index(drop = True).copy()
        copyData(dataset, f"./cache/{rid}/", relativeSize, brightnessDiff, totalSize, [("test", X_test_data)])
        test_probs, test_labels = get_probs_and_labels(best_model, f"./cache/{rid}/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}", "test")
        auc_test = roc_auc_score(test_labels, test_probs)
        print ("Deep, ext AUC:", auc_test)

        return {
            'y_prob': test_probs,
            'y_true': test_labels,
            'auc_int': auc_cv_int,
            'auc_test': auc_test,
            'best_params': []
        }
    else:
        df_train_features, df_test_features = extract_features (df_all, train_ids, test_ids, dataset, method, relativeSize, brightnessDiff, totalSize)

        # apply inner CV
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = optuna.create_study(sampler=optuna.samplers.BruteForceSampler(), direction="maximize")
            # will apply the inner CV too
            study.optimize(lambda trial: inner_objective(trial, dataset, df_train_features, repeat, n_inner_cv))
            best_params = study.best_params
            auc_cv_int = study.best_value

        # refit on all inner data with best parameters, incl. all augmentations
        X_features = df_train_features.drop(columns=["Target", "PatientID"]).columns
        X_train_outer = df_train_features[X_features].values
        y_train_outer = df_train_features["Target"].values

        X_train_outer_selected, fsel, selected_features = select_features(X_train_outer, y_train_outer, best_params['fs_method'], best_params['N'], X_features.tolist())
        clf = getClassifier(best_params)
        clf.fit(X_train_outer_selected, y_train_outer)

        # test model now finally
        X_test_outer = df_test_features.drop(columns=["Target", "PatientID"]).values
        y_test_outer = df_test_features["Target"].values

        X_test_selected = fsel.transform(X_test_outer)
        y_prob = clf.predict_proba(X_test_selected)[:, 1]
        auc_fold = roc_auc_score(y_test_outer, y_prob)
        print ("Rad, AUC on fold", auc_fold)
        return {
            'y_prob': y_prob,
            'y_true': y_test_outer,
            'auc_int': auc_cv_int,
            'auc_test': auc_fold,
            'best_params': best_params,
            'selected_features': selected_features
        }



def nested_cv_optimization(dataset, method, relativeSize, brightnessDiff, totalSize, repeat):
    os.makedirs("./results", exist_ok=True)

    output_path = f"./results/cv_{dataset}_{method}_{relativeSize}_{totalSize}_{brightnessDiff}_{repeat}.pkl"
    if os.path.exists(output_path):
        return None

    print(f"Starting nested CV on dataset {dataset}, with method {method}, size {relativeSize}, totalSize {totalSize}, brightness {brightnessDiff} and repeat {repeat}")

    set_all_seeds(repeat)

    df_all = pd.read_csv(f"./slices/{dataset}_{relativeSize}_{totalSize}_{brightnessDiff}_labels.csv")
    outer_cv = RepeatedStratifiedKFold(n_splits=n_outer_cv, n_repeats=1, random_state=repeat)

    fold_results = []
    for fold_idx, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(df_all["PatientID"], df_all["Target"])):
        train_ids = set(df_all.iloc[outer_train_idx]["PatientID"])
        test_ids = set(df_all.iloc[outer_test_idx]["PatientID"])
        assert train_ids.isdisjoint(test_ids), "Train and test sets are not disjoint!"

        result = evaluate(dataset, method, df_all, train_ids, test_ids, relativeSize, brightnessDiff, totalSize, repeat)
        fold_results.append(result)

    feature_counts = {}
    for r in fold_results:
        selected = r.get('selected_features')
        if selected:
            for f_name in selected:
                feature_counts[f_name] = feature_counts.get(f_name, 0) + 1

    # Sort counts for easier analysis: highest frequency first
    sorted_feature_counts = dict(sorted(feature_counts.items(), key=lambda item: item[1], reverse=True))

    y_true_all = np.concatenate([r['y_true'] for r in fold_results])
    y_prob_all = np.concatenate([r['y_prob'] for r in fold_results])
    auc = roc_auc_score(y_true_all, y_prob_all)

    results = {
        'dataset': dataset,
        'method': method,
        'repeat': repeat,
        'relativeSize': relativeSize,
        'totalSize': totalSize,
        'brightnessDiff': brightnessDiff,
        'auc_test': auc,
        'fold_results': fold_results,
        'feature_selection_frequency': sorted_feature_counts
    }

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    return results



if __name__ == '__main__':
    rmrf("./cache")

    experiments = [(dataset, relativeSize, brightnessDiff, totalSize, repeat)
        for repeat in range(num_repeats)
        for dataset in ["CRLM", "Desmoid", "GIST", "Lipo"]
        for relativeSize in ['different', 'same']
        for brightnessDiff in [15, 33]
        for totalSize in [20, 10]
    ]

    print (f"Executing {len(experiments)} experiments.")

    # do not want to run this in parallel because of deep and joblib doesnt like it
    for (dataset, relativeSize, brightnessDiff, totalSize, repeat) in experiments:
        nested_cv_optimization(dataset, "Radiomics", relativeSize, brightnessDiff, totalSize, repeat)
        nested_cv_optimization(dataset, "Deep", relativeSize, brightnessDiff, totalSize, repeat)
        nested_cv_optimization(dataset, "DeepRadiomics", relativeSize, brightnessDiff, totalSize, repeat)
#
