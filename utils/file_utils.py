# pipeline.py

import os

from murclcsv     import create_murcl_csv
from datasplit    import create_train_val_test_splits
from scratchpaper import combine_csv_files

def run_pipeline(
    train_feat_dir: str,
    train_cluster_dir: str,
    test_feat_dir: str,
    test_cluster_dir: str,
    reference_csv: str,
    intermediate_dir: str,
    base_name: str,
    val_ratio: float,
    random_seed: int,
):
    # -- Step 1: build train CSV
    train_df = create_murcl_csv(
        feature_dir   = train_feat_dir,
        cluster_dir   = train_cluster_dir,
        output_dir    = intermediate_dir,
        output_name   = f"{base_name}_train",
        reference_csv = reference_csv
    )
    train_csv_path = os.path.join(intermediate_dir, f"{base_name}_train.csv")

    # -- Step 2: build test CSV
    test_df = create_murcl_csv(
        feature_dir   = test_feat_dir,
        cluster_dir   = test_cluster_dir,
        output_dir    = intermediate_dir,
        output_name   = f"{base_name}_test",
        reference_csv = reference_csv
    )
    test_csv_path = os.path.join(intermediate_dir, f"{base_name}_test.csv")

    # -- Step 3: split train → train/val, pass-through test
    split_dir = os.path.join(intermediate_dir, "splits")
    create_train_val_test_splits(
        train_input  = train_csv_path,
        test_input   = test_csv_path,
        output_dir   = split_dir,
        output_name  = base_name,
        val_ratio    = val_ratio,
        random_seed  = random_seed
    )

    # -- Step 4: combine train + test CSV into one final CSV
    final_csv = os.path.join(intermediate_dir, f"{base_name}_input.csv")
    combine_csv_files(
        train_csv  = train_csv_path,
        test_csv   = test_csv_path,
        output_csv = final_csv
    )

    print(f"\n✅ Pipeline complete! Final CSV here:\n    {final_csv}")


if __name__ == "__main__":
    # ─── CONFIGURATION (edit these paths as needed) ────────────────────────────────
    # where your .npz/.json feature+cluster files live:
    DIRECTORY           = "C16-SGMuRCL"
    CNN_FEATURE_DIR     = "resnet50"
    TRAIN_FEATURE_DIR   = f"/projects/0/prjs1477/SG-MuRCL/data/{DIRECTORY}/train/features/n"
    TRAIN_CLUSTER_DIR   = f"/projects/0/prjs1477/SG-MuRCL/data/{DIRECTORY}/train/features/"
    TEST_FEATURE_DIR    = f"/projects/0/prjs1477/SG-MuRCL/data/{DIRECTORY}/features/test"
    TEST_CLUSTER_DIR    = f"/projects/0/prjs1477/SG-MuRCL/data/{DIRECTORY}/clusters/test"

    # your reference labels CSV:
    REFERENCE_CSV       = "data/CAMELYON16/evaluation/reference.csv"

    # where everything (train.csv, test.csv, splits/, final CSV) will go:
    INTERMEDIATE_DIR    = "/projects/0/prjs1477/SG-MuRCL/data/{DIRECTORY}"

    # filename prefix (e.g. C16_train.csv, C16_test.csv, splits/C16.json, C16_input.csv)
    BASE_NAME           = "C16"

    # how much of train → val, and RNG seed:
    VAL_RATIO           = 0.2
    RANDOM_SEED         = 985
    # ──────────────────────────────────────────────────────────────────────────────

    run_pipeline(
        train_feat_dir   = TRAIN_FEATURE_DIR,
        train_cluster_dir= TRAIN_CLUSTER_DIR,
        test_feat_dir    = TEST_FEATURE_DIR,
        test_cluster_dir = TEST_CLUSTER_DIR,
        reference_csv    = REFERENCE_CSV,
        intermediate_dir = INTERMEDIATE_DIR,
        base_name        = BASE_NAME,
        val_ratio        = VAL_RATIO,
        random_seed      = RANDOM_SEED,
    )
