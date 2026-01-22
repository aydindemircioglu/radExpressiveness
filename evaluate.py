import shutil
from sklearn.metrics import f1_score
import cv2
from glob import glob
import os
import pickle
from joblib import dump, load
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from collections import defaultdict
from collections import Counter

import nibabel as nib
import nibabel.processing as nibp
import itertools

from PIL import Image
from PIL import ImageDraw, ImageFont

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

n_outer_cv = 5


def readResults ():
    array = np.array
    results = []
    for z in glob(f"./results/cv_*.pkl"):
        with open(z, "rb") as file:
            df = pickle.load(file)

        row = {"Dataset": df["dataset"]}
        row["Repeat"] = df["repeat"]
        row["Method"] = df["method"]
        row["RelativeSize"] = df["relativeSize"]
        row["Brightness"] = df["brightnessDiff"]
        row["TotalSize"] = df["totalSize"]
        assert (len(df["fold_results"]) == n_outer_cv) # outer CV

        # outer test folds
        row["y_prob"] = np.concatenate([df["fold_results"][j]["y_prob"] for j in range(n_outer_cv)])
        row["y_true"] = np.concatenate([df["fold_results"][j]["y_true"] for j in range(n_outer_cv)])
        row["Params"] = np.array([df["fold_results"][j]["best_params"] for j in range(n_outer_cv)])

        # the outer AUC is exactly this
        #auc = roc_auc_score(row["y_true"], row["y_prob"])
        row["AUC"] = df["auc_test"] # this is pooled

        # also extract inner CV at least, but currently not used
        tmp = pd.DataFrame(df["fold_results"])
        row["AUC_Inner"] = tmp["auc_int"].values # this is pooled, but internally
        row["AUC_Outer"] = tmp["auc_test"].values # this is unpooled
        row["AUC_Inner_mean"] = np.mean(tmp["auc_int"].values)
        row["AUC_Outer_mean"] = np.mean(tmp["auc_test"].values)
        results.append(row)

    results = pd.DataFrame(results).reset_index(drop = True)
    return results



def calculate_ci(values, confidence=0.95):
    mean_val = np.mean(values)
    sem = stats.sem(values)  # Standard error of the mean
    ci = sem * stats.t.ppf((1 + confidence) / 2., len(values) - 1)
    return mean_val, mean_val - ci, mean_val + ci


def lighten_color(color, amount=0.4):
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_rgb([(1 - amount) * x + amount for x in c])



def createPlot(results):
    auc_collection = defaultdict(list)
    # Aggregate data
    for _, row in results.iterrows():
        auc_collection[(row['Dataset'], row['Method'], row["RelativeSize"],
                        row["TotalSize"], row["Brightness"])].extend(row['AUC_Outer'])

    df_list = []
    for (dataset, method, relativeSize, totalSize, brightness), all_auc_values in auc_collection.items():
        if len(all_auc_values) > 0:
            # Assuming calculate_ci is defined in your scope
            mean_auc, ci_lower, ci_upper = calculate_ci(np.array(all_auc_values))
            if method == "DeepRadiomics":
                method = "Deep radiomics"
            df_list.append({
                'dataset': dataset,
                'method': method,
                'relativeSize': relativeSize,
                'totalSize': totalSize,
                'brightness': brightness,
                'mean_auc': mean_auc,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            })

    results_df = pd.DataFrame(df_list)

    datasets_sorted = sorted(results_df['dataset'].unique())
    methods = sorted(results_df['method'].unique())

    base_colors = {
        'Radiomics': '#2E86AB',        # Blue
        'Deep radiomics': '#A23B72',   # Purple
        'CNN': '#F18F01'               # Orange
    }

    # Map secondary variables to line styles for distinction
    line_styles = ['-', '--', ':', '-.']

    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']

    # 2x2 Subplots
    fig, axes = plt.subplots(4, 2, figsize=(10, 13), sharey=False)

    y_min = 0.5
    y_max = 1.0


    for idx in range(8):
        ax = axes[idx//2, idx%2]
        axtitle = ''
        if idx > 3:
            subset_df = results_df[results_df['relativeSize'] == 'same'].copy()
            axtitle += "Same total area"
        else:
            subset_df = results_df[results_df['relativeSize'] == 'different'].copy()
            axtitle += "Different total area"

        if idx%4 == 0 or idx%4 == 2:
            subset_df = subset_df[subset_df['totalSize'] == 10].copy()
            axtitle += ", small size"
        else:
            subset_df = subset_df[subset_df['totalSize'] == 20].copy()
            axtitle += ", large size"

        if idx%4 == 0 or idx%4 == 1:
            subset_df = subset_df[subset_df['brightness'] == 15].copy()
            axtitle += ", low visibility"
        else:
            subset_df = subset_df[subset_df['brightness'] == 33].copy()
            axtitle += ", high visibility"


        print (idx)
        print (subset_df)

        ax.set_title(axtitle, fontsize=12, pad=10)

        color = base_colors.get(method, '#333333')

        for i, method in enumerate(methods):
            method_data = subset_df[subset_df['method'] == method].copy()
            if method_data.empty: continue

            x_positions = []
            means = []
            ci_lowers = []
            ci_uppers = []

            base_col = base_colors.get(method, '#333333')
            #current_col = base_col if relativeSize == 'same' else lighten_color(base_col, 0.5)
            label = f"{method}"

            for dataset in datasets_sorted:
                dataset_method_data = method_data[method_data['dataset'] == dataset]
                if not dataset_method_data.empty:
                    x_positions.append(datasets_sorted.index(dataset))
                    means.append(dataset_method_data['mean_auc'].iloc[0])
                    ci_lowers.append(dataset_method_data['ci_lower'].iloc[0])
                    ci_uppers.append(dataset_method_data['ci_upper'].iloc[0])

            if x_positions:  # Only plot if we have data
                # Plot the line
                ax.plot(x_positions, means, 'o-', color=base_col,
                        label=label, linewidth=2, markersize=6)

                # Add confidence interval as shaded area
                ax.fill_between(x_positions, ci_lowers, ci_uppers,
                               color=base_col, alpha=0.2)

        ax.text(-0.05, 1.15, f'({chr(ord("a") + idx)})', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='right')
        ax.set_xlabel('Dataset', fontsize=13)
        ax.set_ylabel('AUC', fontsize=13)
        ax.set_xticks(range(len(datasets_sorted)))
        ax.set_xticklabels(datasets_sorted)#, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.5)
        # ax.legend(frameon=True, fancybox=True, shadow=False,
        #           framealpha=0.9, edgecolor='black')#, linewidth=0.5)
        ax.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 0.47),
                      bbox_transform=ax.get_yaxis_transform(),
                      ncol=len(methods),
                      frameon=True,
                      fontsize=10,
                      edgecolor='black')
        y_min = results_df['ci_lower'].min() * 0.95
        y_max = results_df['ci_upper'].max() * 1.02
        y_min = 0.35
        y_max = 1.02
        ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    os.makedirs('./paper', exist_ok=True)
    plt.savefig('./paper/Figure_3.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    return (results_df.sort_values(["dataset", "method"]))


def getResults():
    # recompute everything
    print("Recomputing AUCs")
    results = readResults ()
    os.makedirs("./paper", exist_ok = True)
    _ = dump(results, "./paper/results_trial.dump")
    return results



def createDatasetTable():
    tbl = []
    for dataset in radMLBench.listDatasets():
        m = radMLBench.getMetaData(dataset)
        _, y = radMLBench.loadData(dataset, return_X_y=True)
        tbl.append({"Dataset": dataset, "Modality": m["modality"], "Outcome": m["outcome"],
            "Instances": m['nInstances'], "Positive Instances": np.sum(y == 1),
            "Negative Instances": np.sum ( y== 0),
            "Features": m["nFeatures"], "Dimensionality": m["Dimensionality"], "Balance": m["ClassBalance"]})
    tbl = pd.DataFrame(tbl)
    tbl.to_excel("./paper/TableDatasets.xlsx")



def save_feature_plot(data, title, filename, color='#2E86AB'):
    if not data:
        return

    feat_df = pd.DataFrame(data).sort_values("Frequency", ascending=False)
    plt.figure(figsize=(10, max(5, len(feat_df) * 0.4)))
    sns.set_theme(style="white")

    sns.barplot(data=feat_df, x="Frequency", y="Feature", color=color)
    plt.axvline(x=50, color='red', linestyle='--', alpha=0.6)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Selection Frequency (%)", fontsize=16)
    plt.ylabel("", fontsize=16)
    plt.xlim(0, 105)
    plt.grid(axis='x', linestyle='-', alpha=0.2)
    # plt.legend()
    # plt.tight_layout()
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



def createFeaturePlots(tbl):
    # 1. Collect all raw data into a flat list
    all_results = []
    files = [z for z in glob("./results/cv_*.pkl") if "_Rad" in z]

    # Pre-parse filenames to make matching faster
    for fpath in files:
        with open(fpath, "rb") as f:
            df = pickle.load(f)

        # Extract params from filename or df
        # (Assuming standard format: ...relativeSize_totalSize_brightnessDiff...)
        for fold in df.get("fold_results", []):
            all_results.append({
                "dataset": df["dataset"],
                "features": fold.get("selected_features", []),
                "filename": fpath,
                "relSize": "same" if "_same_" in fpath else "different"
            })

    results_df = pd.DataFrame(all_results)
    datasets = results_df["dataset"].unique()

    # 2. Loop through parameters to create specific plots
    params = [
        (rel, br, size)
        for rel in ["same", "different"]
        for br in [15, 33]
        for size in [20, 10]
    ]

    for rel, br, size in params:
        token = f"{rel}_{size}_{br}"

        for ds in datasets:
            # Filter for this specific condition
            subset = results_df[
                (results_df["dataset"] == ds) &
                (results_df["filename"].str.contains(token))
            ]

            if subset.empty:
                continue

            # Calculate Frequencies
            flat_features = [item for sublist in subset["features"] for item in sublist]
            total_folds = len(subset)
            counts = Counter(flat_features)

            plot_data = [
                {"Feature": f, "Frequency": (c / total_folds) * 100}
                for f, c in counts.items() if (c / total_folds) > 0.5
            ]

            # Get AUC for title
            sub_tbl = tbl.query('dataset==@ds & method=="Radiomics" & relativeSize==@rel & totalSize==@size & brightness==@br')
            aucperf = np.round(sub_tbl.iloc[0]["mean_auc"], 2) if not sub_tbl.empty else "N/A"

            rel_str = 'Same total area' if rel == 'same' else 'Different total area'
            title = f"{ds} ({rel_str}), AUC: {aucperf}"
            os.makedirs("./paper/features", exist_ok = True)
            fname = f'./paper/features/Features_{ds}_{rel}_{size}_{br}.png'

            save_feature_plot(plot_data, title, fname)


    for relSize in ["same", "different"]:
        for ds in datasets:
            subset = results_df[results_df["dataset"] == ds]
            subset = subset.query("relSize == @relSize").copy()

            flat_features = [item for sublist in subset["features"] for item in sublist]
            total_folds = len(subset)
            counts = Counter(flat_features).most_common(5)

            plot_data = [
                {"Feature": f, "Frequency": (c / total_folds) * 100}
                for f, c in counts
            ]

            title = f"{ds}"
            fname = f'./paper/features/Features_{ds}_{relSize}_agg.png'
            save_feature_plot(plot_data, title, fname, color='#000000')


    datasets_sorted = sorted(datasets)
    def save_grid(rel_type, out_path):
        imgs = [Image.open(f'./paper/features/Features_{ds}_{rel_type}_agg.png') for ds in datasets_sorted]
        w, h = imgs[0].size
        grid = Image.new('RGB', (w * 2, h * 2), 'white')
        grid.paste(imgs[0], (0, 0))
        grid.paste(imgs[1], (w, 0))
        grid.paste(imgs[2], (0, h))
        grid.paste(imgs[3], (w, h))
        grid.save(out_path)
        return out_path

    path_top = save_grid("different", "./paper/features/Figure_4_top.png")
    path_bottom = save_grid("same", "./paper/features/Figure_4_bottom.png")

    # Assemble final figure with Matplotlib
    fig = plt.figure(figsize=(12, 16))
    # Define 2 rows for images, but with height ratios to accommodate the titles
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=-0.15)

    for i, (path, title) in enumerate([(path_top, "Different total area"),
                                       (path_bottom, "Same total area")]):
        ax = fig.add_subplot(gs[i])
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=16, pad=15)
        ax.axis('off')

    plt.savefig('./paper/Figure_4.png', dpi=300, bbox_inches='tight')
    plt.close()

    # image_paths = [f'./paper/features/Features_{ds}_different_agg.png' for ds in sorted(datasets)]
    # image_paths.extend([f'./paper/features/Features_{ds}_same_agg.png' for ds in sorted(datasets)])
    # imgs = [Image.open(img_path) for img_path in image_paths if os.path.exists(img_path)]
    # w, h = imgs[0].size
    # grid_img = Image.new('RGB', (w * 2, h * 4+h//2), color='white')
    # grid_img.paste(imgs[0], (0, 0+h//4))
    # grid_img.paste(imgs[1], (w, 0+h//4))
    # grid_img.paste(imgs[2], (0, h+h//4))
    # grid_img.paste(imgs[3], (w, h+h//4))
    # grid_img.paste(imgs[0], (0, 2*h+h//2))
    # grid_img.paste(imgs[1], (w, 2*h+h//2))
    # grid_img.paste(imgs[2], (0, 3*h+h//2))
    # grid_img.paste(imgs[3], (w, 3*h+h//2))
    # grid_img.save('./paper/Figure_4.png', dpi=(300, 300))
    #

#
# def showBalance():
#     for relativeSize in ["same", "different"]:
#         for brightnessDiff in [15, 33]:
#             for totalSize in [20, 10]:
#                 for dset in ["CRLM", "Desmoid", "GIST", "Lipo"]:
#                     df = pd.read_csv(f"./slices/{dset}_{relativeSize}_{totalSize}_{brightnessDiff}_labels.csv")
#                     print (f"{dset:<16} {np.round(np.sum(df['Target'])/len(df)*100,2)}")
#                 # works out, tested it, so skip it
#                 return None
#

def computeValues(results_df):
    results_df

    rad_mask = results_df['Method'] == 'Radiomics'
    deep_mask = results_df['Method'].isin(['Deep', 'Deep radiomics'])
    results_df.keys()
    rad_low = results_df[rad_mask & (results_df['Brightness'] == 15)]['AUC_Outer_mean']
    rad_high = results_df[rad_mask & (results_df['Brightness'] == 33)]['AUC_Outer_mean']

    deep_all = results_df[deep_mask]['AUC_Outer_mean']

    print("--- Computed AUC Values (IQR reported as range) ---")
    print(f"Radiomics (Low Visibility):  Median {rad_low.median():.2f}, IQR {rad_low.quantile(0.25):.2f}–{rad_low.quantile(0.75):.2f}")
    print(f"Radiomics (High Visibility): Median {rad_high.median():.2f}, IQR {rad_high.quantile(0.25):.2f}–{rad_high.quantile(0.75):.2f}")
    print(f"Deep Models (Overall):       Median {deep_all.median():.2f}, IQR {deep_all.quantile(0.25):.2f}–{deep_all.quantile(0.75):.2f}")


if __name__ == '__main__':
    # not so useful anymore.
    # createDatasetTable()
    results = getResults()
    computeValues (results)
    tbl = createPlot(results)
    createFeaturePlots(tbl)

#
