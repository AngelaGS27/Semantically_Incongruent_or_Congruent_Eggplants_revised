from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_EXCLUDE_COLUMNS = {
    "subject",
    "participant_id",
    "analysis",
    "source_file",
    "source_path",
    "condition",
    "condition_label",
    "before_trial_rejection",
    "after_trial_rejection_reported",
    "after_trial_rejection_mat",
    "trial_type",
    "stim_key",
    "stim_file",
    "stimulus",
    "stimulus_row",
    "sentence_id",
    "item",
    "trial",
    "retained_trial",
    "eeg_trial",
    "epoch_id",
    "subject_trial",
    "condition_trial",
    "experimental_trial",
    "original_event_row",
    "urevent_index",
    "urevent_seconds",
    "event_time_difference_seconds",
    "target_onset_seconds",
    "onset",
    "duration",
    "sample",
    "value",
    "event_id",
    "epoch",
    "epoch_index",
    "channel",
    "channel_index",
    "time",
    "time_index",
    "amplitude",
    "x",
    "y",
    "z",
    "sph_theta",
    "sph_phi",
    "sph_radius",
    "theta",
    "radius",
    "channel_status",
    "channel_status_description",
}


def load_design_matrix(design_path: Path) -> pd.DataFrame:
    design_path = Path(design_path).expanduser().resolve()

    if not design_path.exists():
        raise FileNotFoundError(f"Design matrix not found: {design_path}")

    design = pd.read_csv(design_path, sep=None, engine="python")

    if design.empty:
        raise ValueError(f"Design matrix is empty: {design_path}")

    design = design.copy()

    if "stim_key" not in design.columns:
        raise ValueError("Design matrix does not contain stim_key.")

    if "eeg_trial" not in design.columns:
        raise ValueError("Design matrix does not contain eeg_trial.")

    design["stim_key"] = design["stim_key"].astype("string").str.strip()

    missing_stim_keys = design["stim_key"].isna() | design["stim_key"].eq("")

    if missing_stim_keys.any():
        raise ValueError(
            f"Design matrix contains {int(missing_stim_keys.sum())} missing or empty stim_key values."
        )

    design["eeg_trial"] = pd.to_numeric(design["eeg_trial"], errors="raise").astype(int)

    if design["eeg_trial"].duplicated().any():
        examples = (
            design.loc[
                design["eeg_trial"].duplicated(keep=False),
                "eeg_trial",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Design matrix contains duplicate eeg_trial values. Examples: {examples}"
        )

    design = design.sort_values("eeg_trial", kind="stable").reset_index(drop=True)

    expected_order = np.arange(1, len(design) + 1, dtype=int)
    actual_order = design["eeg_trial"].to_numpy(dtype=int)

    if not np.array_equal(actual_order, expected_order):
        raise ValueError(
            f"eeg_trial must be a complete consecutive sequence from 1 to {len(design)}."
        )

    if {"condition", "retained_trial"}.issubset(design.columns):
        duplicated_trials = design.duplicated(
            subset=["condition", "retained_trial"],
            keep=False,
        )

        if duplicated_trials.any():
            examples = (
                design.loc[
                    duplicated_trials,
                    ["condition", "retained_trial"],
                ]
                .head(10)
                .to_dict("records")
            )

            raise ValueError(
                f"Design matrix contains duplicate retained trials. Examples: {examples}"
            )

    return design


def decode_matlab_string(h5: h5py.File, value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, h5py.Reference):
        if not value:
            return ""

        return decode_matlab_string(
            h5,
            np.asarray(h5[value]).squeeze(),
        )

    arr = np.asarray(value).squeeze()

    if arr.size == 0:
        return ""

    if arr.dtype.kind in {"u", "i"}:
        return "".join(
            chr(int(item))
            for item in np.ravel(arr)
            if int(item) != 0
        ).strip()

    if arr.dtype.kind in {"S", "U"}:
        return "".join(
            str(item)
            for item in np.ravel(arr)
        ).strip()

    if arr.dtype.kind == "O":
        values = [
            decode_matlab_string(
                h5,
                item,
            )
            for item in np.ravel(arr)
        ]

        values = [
            item
            for item in values
            if item
        ]

        if values:
            return values[0]

        return ""

    if arr.size == 1:
        return str(
            arr.item()
        ).strip()

    return str(
        arr
    ).strip()


def read_hdf5_cell_string_vector(h5: h5py.File, dataset) -> list[str]:
    values = np.asarray(
        dataset
    )

    out = []

    for item in values.ravel():
        out.append(
            decode_matlab_string(
                h5,
                item,
            )
        )

    return out


def read_hdf5_cell_numeric_vector(h5: h5py.File, dataset) -> np.ndarray:
    values = np.asarray(
        dataset
    )

    out = []

    for item in values.ravel():
        if isinstance(item, h5py.Reference):
            if item:
                arr = np.asarray(
                    h5[item]
                ).squeeze()
            else:
                arr = np.array(
                    np.nan
                )
        else:
            arr = np.asarray(
                item
            ).squeeze()

        arr = np.asarray(
            arr
        ).astype(
            float
        ).ravel()

        if arr.size == 0:
            out.append(
                np.nan
            )
        else:
            out.append(
                arr[0]
            )

    return np.asarray(
        out,
        dtype=float,
    )


def get_erp_condition_group(
    h5: h5py.File,
    erps_dataset,
    condition_index: int,
):
    reference = erps_dataset[
        0,
        condition_index,
    ]

    return h5[
        reference
    ]


def read_condition_times(
    condition_group,
) -> np.ndarray:
    if "times" not in condition_group:
        raise KeyError(
            "ERP condition does not contain times."
        )

    times = np.asarray(
        condition_group[
            "times"
        ]
    ).squeeze().astype(
        float
    )

    if times.ndim != 1:
        raise ValueError(
            f"ERP times must be one-dimensional, got shape {times.shape}."
        )

    if len(
        times
    ) == 0:
        raise ValueError(
            "ERP times are empty."
        )

    if not np.isfinite(
        times
    ).all():
        raise ValueError(
            "ERP times contain NaN or infinite values."
        )

    return times


def read_condition_channel_names(
    h5: h5py.File,
    condition_group,
    n_channels: int,
) -> list[str]:
    if "chanlocs" not in condition_group:
        raise KeyError(
            "ERP condition does not contain chanlocs."
        )

    chanlocs = condition_group[
        "chanlocs"
    ]

    if "labels" not in chanlocs:
        raise KeyError(
            "ERP condition chanlocs does not contain labels."
        )

    channel_names = read_hdf5_cell_string_vector(
        h5,
        chanlocs[
            "labels"
        ],
    )

    if len(
        channel_names
    ) != n_channels:
        raise ValueError(
            f"Number of channel labels does not match data channels. "
            f"Labels: {len(channel_names)}; channels: {n_channels}."
        )

    missing = [
        index
        for index, channel in enumerate(
            channel_names
        )
        if not str(
            channel
        ).strip()
    ]

    if missing:
        channel_names = [
            str(
                channel
            ).strip()
            if str(
                channel
            ).strip()
            else f"ch_{index + 1:03d}"
            for index, channel in enumerate(
                channel_names
            )
        ]

    return channel_names


def read_condition_eventurevent(
    h5: h5py.File,
    condition_group,
    n_trials: int,
) -> np.ndarray:
    if "epoch" not in condition_group:
        raise KeyError(
            "ERP condition does not contain epoch."
        )

    epoch_group = condition_group[
        "epoch"
    ]

    if "eventurevent" not in epoch_group:
        raise KeyError(
            "ERP condition epoch does not contain eventurevent."
        )

    eventurevent = read_hdf5_cell_numeric_vector(
        h5,
        epoch_group[
            "eventurevent"
        ],
    )

    if len(
        eventurevent
    ) != n_trials:
        raise ValueError(
            f"eventurevent length does not match condition trial count. "
            f"eventurevent: {len(eventurevent)}; trials: {n_trials}."
        )

    if not np.isfinite(
        eventurevent
    ).all():
        raise ValueError(
            "eventurevent contains NaN or infinite values."
        )

    return eventurevent.astype(
        int
    )


def read_epochs_hdf5_mat(mat_path: Path, n_design_rows: int):
    mat_path = Path(
        mat_path
    ).expanduser().resolve()

    if not mat_path.exists():
        raise FileNotFoundError(
            f"EEG .mat file not found: {mat_path}"
        )

    condition_data = []
    condition_trial_counts = []
    eventurevent_values = []
    condition_group_names = []
    channel_names = None
    times = None

    with h5py.File(
        mat_path,
        "r",
    ) as h5:
        if "ERPs" not in h5:
            raise KeyError(
                "ERP MAT file does not contain ERPs."
            )

        erps = h5[
            "ERPs"
        ]

        if erps.ndim != 2:
            raise ValueError(
                f"ERPs must be two-dimensional, got shape {erps.shape}."
            )

        n_conditions = int(
            erps.shape[1]
        )

        if n_conditions == 0:
            raise ValueError(
                "ERPs contains no conditions."
            )

        for condition_index in range(
            n_conditions
        ):
            condition_group = get_erp_condition_group(
                h5=h5,
                erps_dataset=erps,
                condition_index=condition_index,
            )

            condition_group_names.append(
                condition_group.name
            )

            if "data" not in condition_group:
                raise KeyError(
                    f"ERP condition {condition_index + 1} does not contain data."
                )

            raw_data = np.asarray(
                condition_group[
                    "data"
                ],
                dtype=float,
            )

            if raw_data.ndim != 3:
                raise ValueError(
                    f"ERP condition {condition_index + 1} data must be 3D, got shape {raw_data.shape}."
                )

            n_trials = int(
                raw_data.shape[0]
            )

            n_timepoints = int(
                raw_data.shape[1]
            )

            n_channels = int(
                raw_data.shape[2]
            )

            current_times = read_condition_times(
                condition_group
            )

            if len(
                current_times
            ) != n_timepoints:
                raise ValueError(
                    f"Condition {condition_index + 1} time count does not match data. "
                    f"Times: {len(current_times)}; data timepoints: {n_timepoints}."
                )

            current_channel_names = read_condition_channel_names(
                h5=h5,
                condition_group=condition_group,
                n_channels=n_channels,
            )

            if times is None:
                times = current_times
            elif not np.array_equal(
                times,
                current_times,
            ):
                raise ValueError(
                    f"Condition {condition_index + 1} times do not match previous conditions."
                )

            if channel_names is None:
                channel_names = current_channel_names
            elif channel_names != current_channel_names:
                raise ValueError(
                    f"Condition {condition_index + 1} channel labels do not match previous conditions."
                )

            current_eventurevent = read_condition_eventurevent(
                h5=h5,
                condition_group=condition_group,
                n_trials=n_trials,
            )

            condition_data.append(
                np.transpose(
                    raw_data,
                    (
                        0,
                        2,
                        1,
                    ),
                )
            )

            eventurevent_values.append(
                current_eventurevent
            )

            condition_trial_counts.append(
                n_trials
            )

    data = np.concatenate(
        condition_data,
        axis=0,
    )

    eventurevent = np.concatenate(
        eventurevent_values,
        axis=0,
    )

    if data.shape[0] != n_design_rows:
        raise ValueError(
            f"Total ERP trials must match design rows. "
            f"ERP trials: {data.shape[0]}; design rows: {n_design_rows}; "
            f"condition trial counts: {condition_trial_counts}."
        )

    if len(
        eventurevent
    ) != n_design_rows:
        raise ValueError(
            f"Combined eventurevent length must match design rows. "
            f"eventurevent: {len(eventurevent)}; design rows: {n_design_rows}."
        )

    epoch_identity = pd.DataFrame(
        {
            "eeg_trial": np.arange(
                1,
                n_design_rows + 1,
                dtype=int,
            ),
            "eventurevent": eventurevent,
        }
    )

    return {
        "data": data,
        "times": times,
        "channel_names": channel_names,
        "epoch_identity": epoch_identity,
        "hdf5_group": "ERPs",
        "condition_group_names": condition_group_names,
        "condition_trial_counts": condition_trial_counts,
        "raw_data_shape": tuple(
            data.shape
        ),
        "n_conditions": len(
            condition_trial_counts
        ),
    }


def normalise_alignment_values(values) -> pd.Series:
    series = pd.Series(values)

    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        return numeric.astype(int).astype(str)

    return series.astype("string").str.strip()


def find_epoch_alignment_pair(epoch_identity: pd.DataFrame, design: pd.DataFrame) -> tuple[str, str] | None:
    candidate_pairs = [
        ("eventurevent", "urevent_index"),
        ("eeg_trial", "eeg_trial"),
    ]

    for epoch_column, design_column in candidate_pairs:
        if epoch_column in epoch_identity.columns and design_column in design.columns:
            return epoch_column, design_column

    return None


def validate_epoch_alignment(
    n_epochs: int,
    epoch_identity: pd.DataFrame,
    design: pd.DataFrame,
    allow_unverified_alignment: bool = False,
) -> None:
    if len(design) != n_epochs:
        raise ValueError(
            f"Design rows and EEG epochs do not match. Design rows: {len(design)}, EEG epochs: {n_epochs}."
        )

    pair = find_epoch_alignment_pair(epoch_identity=epoch_identity, design=design)

    if pair is None:
        message = (
            "Cannot verify EEG epoch identity against the design matrix. "
            f"Epoch columns available: {epoch_identity.columns.tolist()}. "
            f"Design columns available: {design.columns.tolist()}."
        )

        if allow_unverified_alignment:
            print("WARNING: " + message)
            return

        raise ValueError(message)

    epoch_column, design_column = pair

    epoch_values = normalise_alignment_values(epoch_identity[epoch_column])
    design_values = normalise_alignment_values(design[design_column])

    mismatched = epoch_values.to_numpy() != design_values.to_numpy()

    if mismatched.any():
        mismatch_indices = np.where(mismatched)[0][:10]

        examples = []

        for index in mismatch_indices:
            examples.append(
                {
                    "row": int(index + 1),
                    "epoch_column": epoch_column,
                    "epoch_value": str(epoch_values.iloc[index]),
                    "design_column": design_column,
                    "design_value": str(design_values.iloc[index]),
                    "stim_key": str(design["stim_key"].iloc[index]) if "stim_key" in design.columns else "",
                }
            )

        raise ValueError(
            f"EEG epoch order does not match the design matrix. Compared {epoch_column} with {design_column}. Examples: {examples}"
        )

    print(f"Verified EEG/design alignment using {epoch_column} == {design_column}.")


def parse_predictor_list(value: str | None) -> list[str] | None:
    if value is None:
        return None

    predictors = [item.strip() for item in value.split(",") if item.strip()]

    return predictors or None


def choose_predictor_columns(
    design: pd.DataFrame,
    requested_predictors: list[str] | None,
) -> list[str]:
    if requested_predictors is not None:
        missing = [column for column in requested_predictors if column not in design.columns]

        if missing:
            raise ValueError("Requested predictors are missing from the design matrix: " + ", ".join(missing))

        candidates = list(requested_predictors)

    else:
        candidates = [column for column in design.columns if column not in DEFAULT_EXCLUDE_COLUMNS]

    selected = []
    rejected_non_numeric = []
    rejected_constant = []
    rejected_all_missing = []

    for column in candidates:
        numeric = pd.to_numeric(design[column], errors="coerce")

        non_missing_count = int(numeric.notna().sum())

        if non_missing_count == 0:
            original_non_missing = int(design[column].notna().sum())

            if original_non_missing == 0:
                rejected_all_missing.append(column)
            else:
                rejected_non_numeric.append(column)

            continue

        unique_count = int(numeric.nunique(dropna=True))

        if unique_count < 2:
            rejected_constant.append(column)
            continue

        selected.append(column)

    if requested_predictors is not None:
        rejected_requested = rejected_non_numeric + rejected_constant + rejected_all_missing

        if rejected_requested:
            raise ValueError(
                f"Some explicitly requested predictors are unusable. Non-numeric: {rejected_non_numeric}; constant: {rejected_constant}; all missing: {rejected_all_missing}"
            )

    if not selected:
        raise ValueError("No usable numeric scientific predictors were found.")

    print(f"Selected {len(selected)} predictor columns.")

    if requested_predictors is None:
        if rejected_non_numeric:
            print("Skipped non-numeric candidate columns: " + ", ".join(rejected_non_numeric))

        if rejected_constant:
            print("Skipped constant candidate columns: " + ", ".join(rejected_constant))

        if rejected_all_missing:
            print("Skipped all-missing candidate columns: " + ", ".join(rejected_all_missing))

    return selected


def build_design_array(
    design: pd.DataFrame,
    predictor_columns: list[str],
    add_intercept: bool = True,
):
    missing_columns = [column for column in predictor_columns if column not in design.columns]

    if missing_columns:
        raise ValueError("Predictor columns are missing from the design matrix: " + ", ".join(missing_columns))

    X_df = design[predictor_columns].apply(pd.to_numeric, errors="coerce")

    valid_trial_mask = X_df.notna().all(axis=1).to_numpy(dtype=bool)

    n_complete = int(valid_trial_mask.sum())
    n_removed = int(len(valid_trial_mask) - n_complete)

    if n_complete == 0:
        raise ValueError("No trials have complete values for all selected predictors.")

    if n_removed > 0:
        print(f"Removing {n_removed} EEG trials because at least one selected predictor is missing.")

    X = X_df.loc[valid_trial_mask].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("Design matrix contains infinite predictor values.")

    predictor_names = list(predictor_columns)

    if add_intercept:
        X = np.column_stack([np.ones(X.shape[0], dtype=float), X])
        predictor_names = ["intercept"] + predictor_names

    rank = int(np.linalg.matrix_rank(X))

    if rank < X.shape[1]:
        print(f"Warning: design matrix is rank-deficient. Rank: {rank}; columns: {X.shape[1]}.")

    degrees_of_freedom = X.shape[0] - rank

    if degrees_of_freedom <= 0:
        raise ValueError(
            f"Not enough complete trials for the selected model. Complete trials: {X.shape[0]}; design rank: {rank}; degrees of freedom: {degrees_of_freedom}."
        )

    return X, predictor_names, valid_trial_mask


def fit_mass_univariate_glm(data: np.ndarray, X: np.ndarray):
    data = np.asarray(data, dtype=float)
    X = np.asarray(X, dtype=float)

    if data.ndim != 3:
        raise ValueError(f"EEG data must have shape trials x channels x times, but received {data.shape}.")

    if X.ndim != 2:
        raise ValueError(f"Design matrix X must be two-dimensional, but received {X.shape}.")

    n_trials, n_channels, n_times = data.shape

    if X.shape[0] != n_trials:
        raise ValueError(
            f"Design rows do not match EEG trials after applying the complete-case mask: {X.shape[0]} versus {n_trials}."
        )

    if not np.isfinite(data).all():
        raise ValueError("EEG data contains NaN or infinite values.")

    if not np.isfinite(X).all():
        raise ValueError("Design matrix contains NaN or infinite values.")

    n_predictors = X.shape[1]

    Y = data.reshape(n_trials, n_channels * n_times)

    rank = int(np.linalg.matrix_rank(X))
    degrees_of_freedom = int(n_trials - rank)

    if degrees_of_freedom <= 0:
        raise ValueError(f"The GLM has no residual degrees of freedom. Trials: {n_trials}; rank: {rank}.")

    pinv_X = np.linalg.pinv(X)

    beta_2d = pinv_X @ Y

    fitted_2d = X @ beta_2d

    residuals_2d = Y - fitted_2d

    residual_sum_squares = np.sum(residuals_2d ** 2, axis=0)

    sigma_squared = residual_sum_squares / degrees_of_freedom

    xtx_inverse = np.linalg.pinv(X.T @ X)

    beta_variance_factors = np.diag(xtx_inverse)

    beta_variance_factors = np.maximum(beta_variance_factors, 0.0)

    standard_error_2d = np.sqrt(beta_variance_factors[:, None] * sigma_squared[None, :])

    with np.errstate(divide="ignore", invalid="ignore"):
        t_2d = np.divide(
            beta_2d,
            standard_error_2d,
            out=np.full_like(beta_2d, np.nan, dtype=float),
            where=(standard_error_2d > 0),
        )

    beta = beta_2d.reshape(n_predictors, n_channels, n_times)

    t_values = t_2d.reshape(n_predictors, n_channels, n_times)

    residual_variance = sigma_squared.reshape(n_channels, n_times)

    return {
        "beta": beta,
        "t": t_values,
        "residual_variance": residual_variance,
        "dof": degrees_of_freedom,
        "rank": rank,
        "xtx_inv": xtx_inverse,
    }


def write_string_dataset(h5, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    h5.create_dataset(name, data=np.array(values, dtype=object), dtype=dtype)


def save_hdf5(
    output_path: Path,
    results: dict,
    X: np.ndarray,
    predictor_names: list[str],
    channel_names: list[str],
    times: np.ndarray,
    valid_trial_mask: np.ndarray,
    metadata: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        h5.create_dataset("beta", data=results["beta"], compression="gzip")
        h5.create_dataset("t", data=results["t"], compression="gzip")
        h5.create_dataset("residual_variance", data=results["residual_variance"], compression="gzip")
        h5.create_dataset("design_matrix", data=X, compression="gzip")
        h5.create_dataset("times", data=times)
        h5.create_dataset("valid_trial_mask", data=valid_trial_mask.astype(int))
        h5.create_dataset("xtx_inv", data=results["xtx_inv"])

        h5.attrs["dof"] = results["dof"]
        h5.attrs["rank"] = results["rank"]
        h5.attrs["metadata_json"] = json.dumps(metadata)

        write_string_dataset(h5, "predictor_names", predictor_names)
        write_string_dataset(h5, "channel_names", channel_names)


def save_long_table(
    output_dir: Path,
    beta: np.ndarray,
    t_values: np.ndarray,
    predictor_names: list[str],
    channel_names: list[str],
    times: np.ndarray,
) -> None:
    rows = []

    for pred_idx, predictor in enumerate(predictor_names):
        for ch_idx, channel in enumerate(channel_names):
            rows.append(
                pd.DataFrame(
                    {
                        "predictor": predictor,
                        "channel": channel,
                        "time": times,
                        "beta": beta[pred_idx, ch_idx, :],
                        "t": t_values[pred_idx, ch_idx, :],
                    }
                )
            )

    out = pd.concat(rows, ignore_index=True)

    out_path = output_dir / "erp_glm_beta_t_long.tsv"
    out.to_csv(out_path, sep="\t", index=False)

    print(f"Saved long beta/t table: {out_path}")

def parse_filter_list(value: str | None) -> list[str] | None:
    if value is None:
        return None

    cleaned = str(value).strip()

    if cleaned.upper() == "ALL":
        return None

    values = [
        item.strip()
        for item in cleaned.split(",")
        if item.strip()
    ]

    return values or None


def normalise_subject_id(value: str) -> str:
    value = str(value).strip()

    if value.isdigit():
        return f"sub-{int(value):02d}"

    if not value.startswith("sub-"):
        return f"sub-{value}"

    return value


def get_single_design_value(
    design: pd.DataFrame,
    column: str,
    design_path: Path,
) -> str:
    if column not in design.columns:
        raise ValueError(
            f"Design matrix does not contain {column}: {design_path}"
        )

    values = (
        design[column]
        .astype("string")
        .dropna()
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    unique_values = values.drop_duplicates().tolist()

    if len(unique_values) != 1:
        raise ValueError(
            f"Design matrix must contain exactly one {column}. "
            f"Found {unique_values}: {design_path}"
        )

    return str(unique_values[0])


def subject_matches_filter(
    subject: str,
    requested_subjects: list[str] | None,
) -> bool:
    if requested_subjects is None:
        return True

    normalised_requested = {
        normalise_subject_id(value)
        for value in requested_subjects
    }

    return normalise_subject_id(subject) in normalised_requested


def analysis_matches_filter(
    analysis: str,
    requested_analyses: list[str] | None,
) -> bool:
    if requested_analyses is None:
        return True

    requested = {
        str(value).strip().lower()
        for value in requested_analyses
    }

    return str(analysis).strip().lower() in requested


def resolve_eeg_mat_path(
    design: pd.DataFrame,
    design_path: Path,
    erp_root: Path | None,
) -> Path:
    if "source_path" in design.columns:
        source_paths = (
            design["source_path"]
            .astype("string")
            .dropna()
            .str.strip()
        )

        source_paths = source_paths[
            source_paths.ne("")
        ]

        unique_source_paths = source_paths.drop_duplicates().tolist()

        if len(unique_source_paths) == 1:
            candidate = Path(
                unique_source_paths[0]
            ).expanduser()

            if candidate.exists():
                return candidate.resolve()

    if "source_file" not in design.columns:
        raise ValueError(
            "Cannot resolve ERP .mat path because the design matrix "
            f"does not contain a usable source_path or source_file: {design_path}"
        )

    source_files = (
        design["source_file"]
        .astype("string")
        .dropna()
        .str.strip()
    )

    source_files = source_files[
        source_files.ne("")
    ]

    unique_source_files = source_files.drop_duplicates().tolist()

    if len(unique_source_files) != 1:
        raise ValueError(
            "Cannot resolve ERP .mat path because source_file is not unique "
            f"in design matrix: {design_path}. Values: {unique_source_files}"
        )

    if erp_root is None:
        raise ValueError(
            "The stored source_path does not exist and --erp-root was not provided. "
            f"Design matrix: {design_path}"
        )

    source_file = unique_source_files[0]

    matches = sorted(
        erp_root.rglob(
            source_file
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find ERP .mat file {source_file} under {erp_root}"
        )

    if len(matches) > 1:
        raise ValueError(
            f"More than one ERP .mat file named {source_file} was found under {erp_root}: "
            + "; ".join(str(path) for path in matches)
        )

    return matches[0].resolve()


def make_model_output_dir(
    output_root: Path,
    subject: str,
    analysis: str,
) -> Path:
    return (
        output_root
        / normalise_subject_id(subject)
        / f"erp-{str(analysis).strip()}"
    )


def append_tsv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        sep="\t",
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


def write_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def run_one_erp_glm_model(
    eeg_mat_path: Path,
    design_path: Path,
    output_dir: Path,
    requested_predictors: list[str] | None,
    add_intercept: bool,
    allow_unverified_alignment: bool,
    write_long_table_output: bool,
) -> dict:
    eeg_mat_path = Path(eeg_mat_path).expanduser().resolve()
    design_path = Path(design_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    if not eeg_mat_path.exists():
        raise FileNotFoundError(
            f"EEG .mat file not found: {eeg_mat_path}"
        )

    if not design_path.exists():
        raise FileNotFoundError(
            f"Design matrix not found: {design_path}"
        )

    design = load_design_matrix(design_path)

    subject = get_single_design_value(
        design=design,
        column="subject",
        design_path=design_path,
    )

    analysis = get_single_design_value(
        design=design,
        column="analysis",
        design_path=design_path,
    )

    print()
    print("========================================")
    print(f"Subject: {subject}")
    print(f"Analysis: {analysis}")
    print(f"ERP MAT: {eeg_mat_path}")
    print(f"Design: {design_path}")
    print(f"Output: {output_dir}")
    print("========================================")

    mat_epochs = read_epochs_hdf5_mat(
        eeg_mat_path,
        n_design_rows=len(design),
    )

    data = mat_epochs["data"]
    channel_names = mat_epochs["channel_names"]
    times = mat_epochs["times"]

    print(f"Design rows: {len(design)}")
    print(f"EEG data shape: {data.shape} = trials x channels x times")

    if "hdf5_group" in mat_epochs:
        print(f"HDF5 group: {mat_epochs['hdf5_group']}")

    if "raw_data_shape" in mat_epochs:
        print(f"Raw data shape: {mat_epochs['raw_data_shape']}")

    validate_epoch_alignment(
        n_epochs=data.shape[0],
        epoch_identity=mat_epochs["epoch_identity"],
        design=design,
        allow_unverified_alignment=allow_unverified_alignment,
    )

    predictor_columns = choose_predictor_columns(
        design=design,
        requested_predictors=requested_predictors,
    )

    print("Predictors used:")

    for column in predictor_columns:
        print(f"  - {column}")

    X, predictor_names, valid_trial_mask = build_design_array(
        design=design,
        predictor_columns=predictor_columns,
        add_intercept=add_intercept,
    )

    data_valid = data[
        valid_trial_mask,
        :,
        :,
    ]

    print(f"Complete trials used: {data_valid.shape[0]}")
    print(f"Design matrix shape: {X.shape}")

    results = fit_mass_univariate_glm(
        data=data_valid,
        X=X,
    )

    h5_path = output_dir / "erp_glm_results.h5"

    metadata = {
        "subject": str(subject),
        "analysis": str(analysis),
        "eeg_mat": str(eeg_mat_path),
        "design": str(design_path),
        "n_epochs_original": int(data.shape[0]),
        "n_epochs_used": int(data_valid.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_times": int(data.shape[2]),
        "n_predictors_including_intercept": int(X.shape[1]),
        "rank": int(results["rank"]),
        "dof": int(results["dof"]),
        "add_intercept": bool(add_intercept),
        "alignment_verification": "verified_or_explicitly_allowed_unverified",
        "model_type": "trial-level mass-univariate ERP GLM",
        "official_limo_toolbox": False,
    }

    if "hdf5_group" in mat_epochs:
        metadata["hdf5_group"] = str(mat_epochs["hdf5_group"])

    if "raw_data_shape" in mat_epochs:
        metadata["raw_data_shape"] = list(mat_epochs["raw_data_shape"])

    save_hdf5(
        output_path=h5_path,
        results=results,
        X=X,
        predictor_names=predictor_names,
        channel_names=channel_names,
        times=times,
        valid_trial_mask=valid_trial_mask,
        metadata=metadata,
    )

    np.save(
        output_dir / "beta.npy",
        results["beta"],
    )

    np.save(
        output_dir / "t_values.npy",
        results["t"],
    )

    np.save(
        output_dir / "residual_variance.npy",
        results["residual_variance"],
    )

    pd.DataFrame(
        {
            "predictor": predictor_names,
            "column_index": range(len(predictor_names)),
        }
    ).to_csv(
        output_dir / "predictor_names.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        {
            "channel": channel_names,
            "channel_index": range(len(channel_names)),
        }
    ).to_csv(
        output_dir / "channel_names.tsv",
        sep="\t",
        index=False,
    )

    pd.DataFrame(
        {
            "time": times,
            "time_index": range(len(times)),
        }
    ).to_csv(
        output_dir / "times.tsv",
        sep="\t",
        index=False,
    )

    if write_long_table_output:
        save_long_table(
            output_dir=output_dir,
            beta=results["beta"],
            t_values=results["t"],
            predictor_names=predictor_names,
            channel_names=channel_names,
            times=times,
        )

    summary = {
        "subject": str(subject),
        "analysis": str(analysis),
        "eeg_mat": str(eeg_mat_path),
        "design_matrix": str(design_path),
        "output_dir": str(output_dir),
        "status": "done",
        "n_design_rows": int(len(design)),
        "n_epochs_original": int(data.shape[0]),
        "n_epochs_used": int(data_valid.shape[0]),
        "n_channels": int(data.shape[1]),
        "n_times": int(data.shape[2]),
        "n_predictors_including_intercept": int(X.shape[1]),
        "rank": int(results["rank"]),
        "dof": int(results["dof"]),
        "hdf5_output": str(h5_path),
        "error": "",
    }

    write_json(
        output_dir / "erp_glm_model_summary.json",
        summary,
    )

    print("Done.")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    return summary


def discover_design_matrices(
    design_dir: Path,
) -> list[Path]:
    design_dir = Path(
        design_dir
    ).expanduser().resolve()

    if not design_dir.exists():
        raise FileNotFoundError(
            f"Design directory not found: {design_dir}"
        )

    if not design_dir.is_dir():
        raise NotADirectoryError(
            f"Design path is not a directory: {design_dir}"
        )

    design_paths = sorted(
        design_dir.glob(
            "*_design_matrix.tsv"
        )
    )

    if not design_paths:
        raise FileNotFoundError(
            f"No *_design_matrix.tsv files found in {design_dir}"
        )

    return design_paths

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trial-level mass-univariate ERP GLMs from prepared subject-analysis design matrices."
    )

    parser.add_argument(
        "--eeg-mat",
        default=None,
        help="Single-model mode: path to one epoched ERP .mat file.",
    )

    parser.add_argument(
        "--design",
        default=None,
        help="Single-model mode: path to one subject-analysis design matrix TSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Single-model mode: output directory for one model.",
    )

    parser.add_argument(
        "--design-dir",
        default=None,
        help="Batch mode: directory containing *_design_matrix.tsv files.",
    )

    parser.add_argument(
        "--erp-root",
        default=None,
        help="Batch mode fallback: ERP root used only when source_path inside a design matrix does not exist.",
    )

    parser.add_argument(
        "--output-root",
        default=None,
        help="Batch mode: root directory where ERP GLM outputs will be written.",
    )

    parser.add_argument(
        "--subjects",
        default="ALL",
        help="Batch mode: ALL or comma-separated subject IDs.",
    )

    parser.add_argument(
        "--analyses",
        default="ALL",
        help="Batch mode: ALL or comma-separated ERP analyses.",
    )

    parser.add_argument(
        "--predictor-list",
        default=None,
        help="Optional comma-separated predictors.",
    )

    parser.add_argument(
        "--no-intercept",
        action="store_true",
        help="Do not add intercept column.",
    )

    parser.add_argument(
        "--allow-unverified-alignment",
        action="store_true",
        help="Continue if epoch identity cannot be verified against the design matrix.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip models whose erp_glm_results.h5 already exists.",
    )

    parser.add_argument(
        "--no-long-table",
        action="store_true",
        help="Do not write erp_glm_beta_t_long.tsv.",
    )

    args = parser.parse_args()

    requested_predictors = parse_predictor_list(args.predictor_list)

    add_intercept = not args.no_intercept

    write_long_table_output = not args.no_long_table

    single_mode_values = [
        args.eeg_mat,
        args.design,
        args.output_dir,
    ]

    single_mode_requested = any(
        value is not None
        for value in single_mode_values
    )

    if single_mode_requested:
        if not all(
            value is not None
            for value in single_mode_values
        ):
            raise ValueError(
                "Single-model mode requires --eeg-mat, --design, and --output-dir."
            )

        run_one_erp_glm_model(
            eeg_mat_path=Path(args.eeg_mat),
            design_path=Path(args.design),
            output_dir=Path(args.output_dir),
            requested_predictors=requested_predictors,
            add_intercept=add_intercept,
            allow_unverified_alignment=args.allow_unverified_alignment,
            write_long_table_output=write_long_table_output,
        )

        return

    if args.design_dir is None:
        raise ValueError(
            "Batch mode requires --design-dir."
        )

    if args.output_root is None:
        raise ValueError(
            "Batch mode requires --output-root."
        )

    design_dir = Path(args.design_dir).expanduser().resolve()

    output_root = Path(args.output_root).expanduser().resolve()

    erp_root = (
        Path(args.erp_root).expanduser().resolve()
        if args.erp_root is not None
        else None
    )

    if erp_root is not None:
        if not erp_root.exists():
            raise FileNotFoundError(
                f"ERP root not found: {erp_root}"
            )

        if not erp_root.is_dir():
            raise NotADirectoryError(
                f"ERP root is not a directory: {erp_root}"
            )

    requested_subjects = parse_filter_list(args.subjects)

    requested_analyses = parse_filter_list(args.analyses)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_config = {
        "mode": "batch",
        "design_dir": str(design_dir),
        "erp_root": str(erp_root) if erp_root is not None else "",
        "output_root": str(output_root),
        "subjects": args.subjects,
        "analyses": args.analyses,
        "predictor_list": requested_predictors,
        "add_intercept": bool(add_intercept),
        "allow_unverified_alignment": bool(args.allow_unverified_alignment),
        "skip_existing": bool(args.skip_existing),
        "write_long_table": bool(write_long_table_output),
        "model_type": "trial-level mass-univariate ERP GLM",
        "official_limo_toolbox": False,
    }

    write_json(
        output_root / "erp_glm_run_config.json",
        run_config,
    )

    batch_summary_path = output_root / "erp_glm_batch_summary.tsv"
    failed_models_path = output_root / "erp_glm_failed_models.tsv"

    if batch_summary_path.exists():
        batch_summary_path.unlink()

    if failed_models_path.exists():
        failed_models_path.unlink()

    design_paths = discover_design_matrices(design_dir)

    print(f"Found {len(design_paths)} design matrices.")
    print(f"Design directory: {design_dir}")
    print(f"Output root: {output_root}")

    if erp_root is not None:
        print(f"ERP root fallback: {erp_root}")

    attempted = 0
    completed = 0
    skipped = 0
    failed = 0

    for design_path in design_paths:
        try:
            design = load_design_matrix(design_path)

            subject = get_single_design_value(
                design=design,
                column="subject",
                design_path=design_path,
            )

            analysis = get_single_design_value(
                design=design,
                column="analysis",
                design_path=design_path,
            )

            if not subject_matches_filter(
                subject=subject,
                requested_subjects=requested_subjects,
            ):
                continue

            if not analysis_matches_filter(
                analysis=analysis,
                requested_analyses=requested_analyses,
            ):
                continue

            attempted += 1

            eeg_mat_path = resolve_eeg_mat_path(
                design=design,
                design_path=design_path,
                erp_root=erp_root,
            )

            output_dir = make_model_output_dir(
                output_root=output_root,
                subject=subject,
                analysis=analysis,
            )

            h5_path = output_dir / "erp_glm_results.h5"

            if args.skip_existing and h5_path.exists():
                row = {
                    "subject": str(subject),
                    "analysis": str(analysis),
                    "eeg_mat": str(eeg_mat_path),
                    "design_matrix": str(design_path),
                    "output_dir": str(output_dir),
                    "status": "skipped",
                    "n_design_rows": int(len(design)),
                    "n_epochs_original": "",
                    "n_epochs_used": "",
                    "n_channels": "",
                    "n_times": "",
                    "n_predictors_including_intercept": "",
                    "rank": "",
                    "dof": "",
                    "hdf5_output": str(h5_path),
                    "error": "",
                }

                append_tsv(
                    pd.DataFrame([row]),
                    batch_summary_path,
                )

                skipped += 1

                print()
                print(f"SKIP existing: {subject} {analysis}")

                continue

            summary = run_one_erp_glm_model(
                eeg_mat_path=eeg_mat_path,
                design_path=design_path,
                output_dir=output_dir,
                requested_predictors=requested_predictors,
                add_intercept=add_intercept,
                allow_unverified_alignment=args.allow_unverified_alignment,
                write_long_table_output=write_long_table_output,
            )

            append_tsv(
                pd.DataFrame([summary]),
                batch_summary_path,
            )

            completed += 1

        except Exception as error:
            failed += 1

            error_message = str(error)

            failed_row = {
                "subject": "",
                "analysis": "",
                "eeg_mat": "",
                "design_matrix": str(design_path),
                "output_dir": "",
                "status": "failed",
                "n_design_rows": "",
                "n_epochs_original": "",
                "n_epochs_used": "",
                "n_channels": "",
                "n_times": "",
                "n_predictors_including_intercept": "",
                "rank": "",
                "dof": "",
                "hdf5_output": "",
                "error": error_message,
            }

            try:
                failed_design = load_design_matrix(design_path)

                failed_row["subject"] = get_single_design_value(
                    design=failed_design,
                    column="subject",
                    design_path=design_path,
                )

                failed_row["analysis"] = get_single_design_value(
                    design=failed_design,
                    column="analysis",
                    design_path=design_path,
                )

                failed_output_dir = make_model_output_dir(
                    output_root=output_root,
                    subject=failed_row["subject"],
                    analysis=failed_row["analysis"],
                )

                failed_row["output_dir"] = str(failed_output_dir)

                try:
                    failed_row["eeg_mat"] = str(
                        resolve_eeg_mat_path(
                            design=failed_design,
                            design_path=design_path,
                            erp_root=erp_root,
                        )
                    )

                except Exception:
                    failed_row["eeg_mat"] = ""

            except Exception:
                pass

            append_tsv(
                pd.DataFrame([failed_row]),
                batch_summary_path,
            )

            append_tsv(
                pd.DataFrame([failed_row]),
                failed_models_path,
            )

            print()
            print(f"FAILED: {design_path}")
            print(error_message)

    final_summary = {
        "attempted": int(attempted),
        "completed": int(completed),
        "skipped": int(skipped),
        "failed": int(failed),
        "batch_summary": str(batch_summary_path),
        "failed_models": str(failed_models_path),
        "run_config": str(output_root / "erp_glm_run_config.json"),
    }

    write_json(
        output_root / "erp_glm_batch_summary.json",
        final_summary,
    )

    print()
    print("Batch complete.")
    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )

    if attempted == 0:
        raise RuntimeError(
            "No design matrices matched the requested subjects/analyses."
        )

    if completed == 0 and skipped == 0:
        raise RuntimeError(
            "No ERP GLM models completed successfully."
        )


if __name__ == "__main__":
    main()