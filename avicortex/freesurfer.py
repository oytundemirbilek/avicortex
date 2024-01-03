"""Module for freesurfer utility functions."""
# import os

# import nibabel as nib
# import numpy as np
# import pandas as pd
# from nibabel.freesurfer.io import read_annot, read_label
# from nibabel.nifti1 import Nifti1Image

# # Sanity check params
# ROOT_PATH = os.path.dirname(__file__)


# def find_missing_regions_in_segmentations(DATASET_FILE, subject=None):
#     if subject is None:
#         all_subjects = os.listdir(os.path.join(ROOT_PATH, DATASET_FILE))
#         subject = all_subjects[0]
#         if subject == "logs":
#             subject = all_subjects[1]
#     org_path = os.path.join(ROOT_PATH, DATASET_FILE, subject, "mri", "orig.mgz")
#     seg_path = os.path.join(
#         ROOT_PATH, DATASET_FILE, subject, "mri", "aparc.DKTatlas+aseg.deep.mgz"
#     )

#     org = nib.load(org_path)
#     seg = nib.load(seg_path)
#     print(seg.get_fdata())
#     regions_found = np.unique(seg.get_fdata())
#     print(regions_found)
#     print(len(regions_found))


# def convert_to_nifti(DATASET_FILE, subject=None):
#     if subject is None:
#         all_subjects = os.listdir(os.path.join(ROOT_PATH, DATASET_FILE))
#         subject = all_subjects[0]
#         if subject == "logs" or subject == "fsaverage":
#             subject = all_subjects[1]

#     org_path = os.path.join(ROOT_PATH, DATASET_FILE, subject, "mri", "orig.mgz")
#     seg_path = os.path.join(
#         ROOT_PATH, DATASET_FILE, subject, "mri", "aparc.DKTatlas+aseg.deep.mgz"
#     )

#     org = nib.load(org_path)
#     seg = nib.load(seg_path)

#     if not isinstance(org, Nifti1Image):
#         org = Nifti1Image(org.get_fdata(), org.affine, header=nib.Nifti1Header())

#     if not isinstance(seg, Nifti1Image):
#         seg = Nifti1Image(seg.get_fdata(), seg.affine, header=nib.Nifti1Header())

#     nib.save(org, f"{ROOT_PATH}/orig.nii.gz")
#     nib.save(seg, f"{ROOT_PATH}/aseg.nii.gz")


# def sanity_check(DATASET_FOLDER):
#     subjects = os.listdir(os.path.join(ROOT_PATH, DATASET_FOLDER))
#     for sub in subjects:
#         subject_results = os.listdir(os.path.join(ROOT_PATH, DATASET_FOLDER, sub))
#         if "stats" not in subject_results:
#             print("WARNING: No stats found for subject:", sub)
#             continue

#         stat_folder = os.listdir(os.path.join(ROOT_PATH, DATASET_FOLDER, sub, "stats"))
#         if len(stat_folder) == 23:
#             print("SUCCESS: subject:", sub)
#         elif len(stat_folder) < 23:
#             print("WARNING: Missing stats in subject:", sub)
#         else:
#             print("WARNING: More stats then expected in subject:", sub)


# def find_missing_regions(labels):
#     regions = set(range(1, 36))
#     labels = set(labels)
#     missing_regions = regions.difference(labels)
#     print(missing_regions)

#     subject_label_root_path = os.path.join(
#         os.path.dirname(__file__),
#         "nitrc_schiz_fastsurfer_results",
#         "HC_005_MR",
#         "label",
#     )

#     annot_file = "lh.aparc.DKTatlas.mapped.annot"
#     label_file = "lh.cortex.label"
#     # annot_file = "lh.aparc.DKTatlas.annot"

#     annot_path = os.path.join(subject_label_root_path, annot_file)
#     label_path = os.path.join(subject_label_root_path, label_file)
#     # annot_path = os.path.join(subject_label_root_path, annot_file)

#     labels, ctab, names = read_annot(annot_path)
#     label_arr, scalar_arr = read_label(label_path)
#     # labels, ctab, names = read_annot(annot_path)

#     # labels = np.array(labels)
#     # print(len(np.unique(labels)))
#     # print(len(labels))

#     print(label_arr)
#     print(scalar_arr)
#     print(len(np.unique(label_arr)))
#     print(len(label_arr))


# def collect_all_stats_in_table(result_path):
#     result_dir = os.path.join(ROOT_PATH, result_path)
#     stats_dir = os.listdir(result_dir)
#     try:
#         stats_dir.remove("all_stats.csv")
#     except:
#         print("Dir is already clean.")
#     all_stat_tables = []
#     for idx, stats_file in enumerate(stats_dir):
#         stats_df = pd.read_csv(os.path.join(result_dir, stats_file), sep="\t")
#         # stats_df.add_prefix(stats_file.strip("txt."))
#         col_name = stats_df.columns[0]
#         if idx != 0:
#             stats_df = stats_df.drop(col_name, axis=1)
#         else:
#             stats_df = stats_df.rename(columns={col_name: "Subject ID"})
#         all_stat_tables.append(stats_df)
#         print("ADDED:", stats_file)
#     all_df = pd.concat(all_stat_tables, axis=1, join="outer")
#     all_df.to_csv(os.path.join(result_dir, "all_stats.csv"), index=False)


# if __name__ == "__main__":
#     SCHIZ_DATASET_OUT = "nitrc_schiz_fastsurfer_results"
#     CANN_BL_DATASET_OUT = "openneuro_cannabis_baseline_fastsurfer_results"
#     CANN_BL_DATASET_OUT = "openneuro_baseline_freesurfer_results"
#     CANN_FU_DATASET_OUT = "openneuro_cannabis_followup_fastsurfer_results"
#     # view_segmentations("sub-202")
#     # convert_to_nifti(CANN_BL_DATASET_OUT)
#     # sanity_check(CANN_BL_DATASET_OUT)

#     # collect stats params
#     DATASET = "openneuro_destrieuxatlas"
#     TIME = "baseline"
#     collect_all_stats_in_table(os.path.join(ROOT_PATH, "fs_stats", DATASET, TIME))
