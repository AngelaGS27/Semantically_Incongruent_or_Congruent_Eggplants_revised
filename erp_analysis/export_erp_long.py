from pathlib import Path
import argparse
import h5py
import numpy as np
import pandas as pd



def decode_matlab_string(file, ref):
    """Decode MATLAB HDF5 string reference."""
    try:
        obj = file[ref]
        arr = np.array(obj).squeeze()

        if arr.dtype.kind in {"u", "i"}:
            return "".join(chr(int(x)) for x in arr if int(x) != 0)

        return str(arr)

    except Exception:
        return ""

def dereference_numeric_cell(
    file: h5py.File,
    dataset,
    row_index: int,
) -> np.ndarray:
    """
    Dereference one MATLAB HDF5 cell containing numeric values.
    """

    reference = dataset[row_index, 0]
    values = np.asarray(
        file[reference]
    ).squeeze()

    return np.atleast_1d(
        values
    ).astype(float)

def get_subject_id(path: Path) -> str:
    """
    Extract the subject identifier from an ERP filename.
    """

    name = path.name

    if "_task-" in name:
        return name.split(
            "_task-",
            1,
        )[0]

    return path.stem


def get_analysis_name(path: Path) -> str:
    """
    Extract the ERP analysis name from the filename.

    Examples:
        sub-24_task-N400Stimset_erp-CP.mat
            -> CP

        sub-24_task-N400Stimset_erp-GA.mat
            -> GA

        sub-24_task-N400Stimset_erp-LD.mat
            -> LD

        sub-24_task-N400Stimset_erp-Order.mat
            -> Order

        sub-24_task-N400Stimset_erp-Time.mat
            -> Time
    """

    marker = "_erp-"

    if marker not in path.stem:
        raise ValueError(
            "Cannot identify ERP analysis from filename: "
            f"{path.name}"
        )

    return path.stem.split(
        marker,
        1,
    )[1]


def extract_condition_object(
    file,
    erps_dataset,
    condition_index,
):
    """
    Follow one ERP condition object reference.
    """

    reference = erps_dataset[
        0,
        condition_index,
    ]

    return file[
        reference
    ]


def extract_data(
    condition_group,
):
    """
    Extract ERP data.

    Expected shape:
        trials x timepoints x channels
    """

    if "data" not in condition_group:
        raise KeyError(
            "No 'data' field found in ERP condition."
        )

    data = np.asarray(
        condition_group[
            "data"
        ]
    )

    if data.ndim != 3:
        raise ValueError(
            "Expected three-dimensional ERP data, "
            f"but found shape {data.shape}."
        )

    n_trials = data.shape[0]
    n_timepoints = data.shape[1]
    n_channels = data.shape[2]

    return (
        data,
        n_trials,
        n_timepoints,
        n_channels,
    )


def extract_epoch_urevent_indices(
    file: h5py.File,
    condition_group,
    n_trials: int,
) -> list[int]:
    """
    Extract the original EEGLAB urevent index for every
    retained ERP epoch.

    The resulting indices preserve the retained epoch order
    stored inside the ERP MAT file.
    """

    if "epoch" not in condition_group:
        raise KeyError(
            "No epoch structure found in ERP condition."
        )

    epoch_group = condition_group[
        "epoch"
    ]

    if "eventurevent" not in epoch_group:
        raise KeyError(
            "No epoch/eventurevent field found in ERP condition."
        )

    eventurevent = epoch_group[
        "eventurevent"
    ]

    if eventurevent.shape[0] != n_trials:
        raise ValueError(
            "eventurevent row count does not match retained "
            f"trials: {eventurevent.shape[0]} versus {n_trials}"
        )

    indices = []

    for trial_index in range(
        n_trials
    ):
        values = dereference_numeric_cell(
            file=file,
            dataset=eventurevent,
            row_index=trial_index,
        )

        finite_values = values[
            np.isfinite(
                values
            )
        ]

        if len(
            finite_values
        ) == 0:
            raise ValueError(
                "No valid urevent index for retained trial "
                f"{trial_index + 1}."
            )

        urevent_index = int(
            round(
                float(
                    finite_values[0]
                )
            )
        )

        if urevent_index < 1:
            raise ValueError(
                "Invalid EEGLAB urevent index: "
                f"{urevent_index}"
            )

        indices.append(
            urevent_index
        )

    return indices


def extract_times(
    file,
):
    """
    Extract the ERP time vector.

    Converts milliseconds to seconds when the stored values
    are clearly in milliseconds.
    """

    if "t" not in file:
        raise KeyError(
            "No time vector 't' found in ERP MAT file."
        )

    times = np.asarray(
        file[
            "t"
        ]
    ).squeeze().astype(
        float
    )

    if times.ndim != 1:
        raise ValueError(
            "ERP time vector is not one-dimensional: "
            f"{times.shape}"
        )

    if len(
        times
    ) == 0:
        raise ValueError(
            "ERP time vector is empty."
        )

    if not np.isfinite(
        times
    ).all():
        raise ValueError(
            "ERP time vector contains missing or infinite values."
        )

    if np.nanmax(
        np.abs(
            times
        )
    ) > 10:
        times = (
            times
            / 1000.0
        )

    return times


def make_biosemi_128_label(
    index: int,
) -> str:
    """
    Convert a zero-based channel index to a BioSemi 128 label.

    Examples:
        0   -> A1
        31  -> A32
        32  -> B1
        63  -> B32
        64  -> C1
        95  -> C32
        96  -> D1
        127 -> D32
    """

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    if (
        index < 0
        or index >= 128
    ):
        return (
            f"ch_{index + 1:03d}"
        )

    letter = letters[
        index // 32
    ]

    number = (
        index % 32
    ) + 1

    return (
        f"{letter}{number}"
    )


def extract_channel_labels(
    file,
    condition_group,
    n_channels,
):
    """
    Extract channel labels from the ERP MAT file.

    If labels cannot be read, use BioSemi 128 acquisition
    labels in channel order.
    """

    labels = []

    try:
        chanlocs = condition_group[
            "chanlocs"
        ]

        if "labels" in chanlocs:
            label_references = chanlocs[
                "labels"
            ]

            for channel_index in range(
                label_references.shape[0]
            ):
                reference = label_references[
                    channel_index,
                    0,
                ]

                labels.append(
                    decode_matlab_string(
                        file,
                        reference,
                    )
                )

    except Exception:
        labels = []

    invalid_labels = (
        len(
            labels
        ) != n_channels
        or any(
            not str(
                label
            ).strip()
            for label in labels
        )
    )

    if invalid_labels:
        labels = [
            make_biosemi_128_label(
                channel_index
            )
            for channel_index in range(
                n_channels
            )
        ]

    return labels


def clean_electrode_name(
    value,
):
    """
    Clean an electrode label.
    """

    value = (
        str(
            value
        )
        .strip()
        .strip("'")
        .strip('"')
    )

    if "_" in value:
        value = value.split(
            "_",
            1,
        )[0]

    return value.strip()


def split_channel_name(
    channel_name,
):
    """
    Split labels such as A1_Cz into:

        electrode = A1
        standard_label = Cz

    Labels without an underscore retain only the acquisition
    electrode name.
    """

    channel_name = str(
        channel_name
    ).strip()

    if "_" in channel_name:
        electrode, standard_label = channel_name.split(
            "_",
            1,
        )

        return (
            clean_electrode_name(
                electrode
            ),
            standard_label.strip(),
        )

    return (
        clean_electrode_name(
            channel_name
        ),
        "",
    )


def build_channel_metadata(
    channel_labels: list[str],
) -> pd.DataFrame:
    """
    Build channel metadata directly from labels stored in the
    ERP MAT file.

    No external channels.tsv, electrodes.tsv, or BIDS directory
    is required.
    """

    rows = []

    for (
        channel_index,
        channel_label,
    ) in enumerate(
        channel_labels
    ):
        channel_label = str(
            channel_label
        ).strip()

        if (
            not channel_label
            or channel_label.lower().startswith(
                "ch_"
            )
        ):
            channel_label = make_biosemi_128_label(
                channel_index
            )

        (
            electrode,
            standard_label,
        ) = split_channel_name(
            channel_label
        )

        channel = (
            standard_label
            if standard_label
            else electrode
        )

        rows.append(
            {
                "channel_index": channel_index,
                "mat_channel": channel_label,
                "channel": channel,
                "original_channel": channel_label,
                "electrode": electrode,
                "standard_label": standard_label,
                "x": np.nan,
                "y": np.nan,
                "z": np.nan,
                "sph_theta": np.nan,
                "sph_phi": np.nan,
                "sph_radius": np.nan,
                "theta": np.nan,
                "radius": np.nan,
                "type": "EEG",
                "units": np.nan,
                "status": np.nan,
                "status_description": np.nan,
            }
        )

    channel_metadata = pd.DataFrame(
        rows
    )

    if len(
        channel_metadata
    ) != len(
        channel_labels
    ):
        raise ValueError(
            "Channel metadata count does not match "
            "the number of channel labels."
        )

    return channel_metadata


def get_trial_rejection_path(
    mat_path: Path,
) -> Path:
    """
    Construct the exact paired trial-rejection TSV path.

    Examples:

        sub-24_task-N400Stimset_erp-CP.mat
        sub-24_task-N400Stimset_erp-CP_trialrej.tsv

        sub-24_task-N400Stimset_erp-GA.mat
        sub-24_task-N400Stimset_erp-GA_trialrej.tsv

        sub-24_task-N400Stimset_erp-LD.mat
        sub-24_task-N400Stimset_erp-LD_trialrej.tsv

        sub-24_task-N400Stimset_erp-Order.mat
        sub-24_task-N400Stimset_erp-Order_trialrej.tsv

        sub-24_task-N400Stimset_erp-Time.mat
        sub-24_task-N400Stimset_erp-Time_trialrej.tsv
    """

    return mat_path.with_name(
        f"{mat_path.stem}_trialrej.tsv"
    )


def load_trial_rejection_summary(
    mat_path: Path,
) -> pd.DataFrame:
    """
    Load the trial-rejection TSV paired with one ERP MAT file.

    Required columns:

        condition
        before_trial_rejection
        after_trial_rejection

    An optional '#' column is used as the condition index when
    present.
    """

    rejection_path = get_trial_rejection_path(
        mat_path
    )

    if not rejection_path.exists():
        raise FileNotFoundError(
            "Trial-rejection TSV not found for ERP MAT file:\n"
            f"MAT: {mat_path}\n"
            f"Expected TSV: {rejection_path}"
        )

    if not rejection_path.is_file():
        raise FileNotFoundError(
            "Trial-rejection path is not a file:\n"
            f"{rejection_path}"
        )

    rejection = pd.read_csv(
        rejection_path,
        sep="\t",
    )

    required_columns = [
        "condition",
        "before_trial_rejection",
        "after_trial_rejection",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rejection.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{rejection_path} is missing required columns: "
            f"{missing_columns}"
        )

    rejection = rejection.copy()

    rejection["condition"] = (
        rejection[
            "condition"
        ]
        .astype(
            "string"
        )
        .str.strip()
    )

    missing_condition = (
        rejection[
            "condition"
        ].isna()
        | rejection[
            "condition"
        ].eq(
            ""
        )
    )

    if missing_condition.any():
        raise ValueError(
            f"{rejection_path} contains missing or empty "
            "condition labels."
        )

    if "#" in rejection.columns:
        rejection[
            "condition_index"
        ] = pd.to_numeric(
            rejection[
                "#"
            ],
            errors="raise",
        ).astype(
            int
        )
    else:
        rejection[
            "condition_index"
        ] = range(
            1,
            len(
                rejection
            ) + 1,
        )

    for column in [
        "before_trial_rejection",
        "after_trial_rejection",
    ]:
        rejection[
            column
        ] = pd.to_numeric(
            rejection[
                column
            ],
            errors="raise",
        ).astype(
            int
        )

        if (
            rejection[
                column
            ] < 0
        ).any():
            raise ValueError(
                f"{rejection_path} contains negative values "
                f"in {column}."
            )

    impossible_counts = (
        rejection[
            "after_trial_rejection"
        ]
        > rejection[
            "before_trial_rejection"
        ]
    )

    if impossible_counts.any():
        examples = (
            rejection.loc[
                impossible_counts,
                [
                    "condition",
                    "before_trial_rejection",
                    "after_trial_rejection",
                ],
            ]
            .head(
                10
            )
            .to_dict(
                "records"
            )
        )

        raise ValueError(
            "Some after_trial_rejection counts are larger than "
            "their before_trial_rejection counts. "
            f"Examples: {examples}"
        )

    rejection[
        "analysis"
    ] = get_analysis_name(
        mat_path
    )

    rejection[
        "trial_rejection_file"
    ] = str(
        rejection_path
    )

    return rejection

def normalise_stim_key(
    series: pd.Series,
) -> pd.Series:
    """
    Normalise stimulus identifiers before matching.
    """

    return (
        series
        .astype("string")
        .str.strip()
    )


def normalise_stim_file(
    series: pd.Series,
) -> pd.Series:
    """
    Normalise stimulus filenames before matching.
    """

    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            "\\",
            "/",
            regex=False,
        )
        .str.rsplit(
            "/",
            n=1,
        )
        .str[-1]
    )


def infer_trial_type_from_stim_file(series: pd.Series) -> pd.Series:
    values = normalise_stim_file(series)
    upper_values = values.astype("string").str.upper()

    trial_type = pd.Series("", index=values.index, dtype="string")
    trial_type = trial_type.mask(upper_values.str.contains("NPC", na=False), "NPC")
    trial_type = trial_type.mask(upper_values.str.contains("NPI", na=False), "NPI")

    return trial_type


def load_stimulus_lookup(
    language_metrics_path: Path,
) -> pd.DataFrame:
    language_metrics_path = Path(
        language_metrics_path
    ).expanduser().resolve()

    if not language_metrics_path.exists():
        raise FileNotFoundError(
            "Language metrics file not found: "
            f"{language_metrics_path}"
        )

    if not language_metrics_path.is_file():
        raise FileNotFoundError(
            "Language metrics path is not a file: "
            f"{language_metrics_path}"
        )

    metrics = pd.read_csv(
        language_metrics_path,
        sep="\t",
    )

    required_columns = [
        "stim_file",
        "stim_key",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in metrics.columns
    ]

    if missing_columns:
        raise ValueError(
            "Language metrics table is missing required columns: "
            f"{missing_columns}. Use ALL_language_metrics.tsv."
        )

    lookup = metrics[
        [
            "stim_file",
            "stim_key",
        ]
    ].copy()

    lookup["original_event_row"] = np.arange(
        1,
        len(lookup) + 1,
        dtype=int,
    )

    lookup["stim_file"] = normalise_stim_file(
        lookup["stim_file"]
    )

    lookup["stim_key"] = normalise_stim_key(
        lookup["stim_key"]
    )

    lookup["trial_type"] = infer_trial_type_from_stim_file(
        lookup["stim_file"]
    )

    missing_stim_files = (
        lookup["stim_file"].isna()
        | lookup["stim_file"].eq("")
    )

    if missing_stim_files.any():
        raise ValueError(
            "Language metrics table contains "
            f"{int(missing_stim_files.sum())} missing or empty "
            "stim_file values."
        )

    missing_stim_keys = (
        lookup["stim_key"].isna()
        | lookup["stim_key"].eq("")
    )

    if missing_stim_keys.any():
        raise ValueError(
            "Language metrics table contains "
            f"{int(missing_stim_keys.sum())} missing or empty "
            "stim_key values."
        )

    missing_trial_type = (
        lookup["trial_type"].isna()
        | lookup["trial_type"].eq("")
    )

    if missing_trial_type.any():
        examples = (
            lookup.loc[
                missing_trial_type,
                "stim_file",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Could not infer NPC/NPI trial_type from some stim_file values. "
            f"Examples: {examples}"
        )

    duplicated_event_rows = lookup["original_event_row"].duplicated(
        keep=False
    )

    if duplicated_event_rows.any():
        raise ValueError(
            "Language metrics table produced duplicate original_event_row values."
        )

    conflicting = (
        lookup
        .groupby(
            "stim_file",
            dropna=False,
        )["stim_key"]
        .nunique(
            dropna=False
        )
    )

    conflicting = conflicting[
        conflicting > 1
    ]

    if not conflicting.empty:
        raise ValueError(
            "Some stim_file values map to more than one stim_key. "
            f"Examples: {conflicting.index[:10].tolist()}"
        )

    print(
        f"Loaded {len(lookup)} stimulus mappings from "
        f"{language_metrics_path}"
    )

    return lookup

def decode_matlab_value(file, value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, h5py.Reference):
        if not value:
            return ""

        obj = file[value]
        arr = np.asarray(obj).squeeze()
        return decode_matlab_value(file, arr)

    arr = np.asarray(value).squeeze()

    if arr.dtype.kind in {"u", "i"} and arr.size > 1:
        return "".join(chr(int(x)) for x in np.ravel(arr) if int(x) != 0).strip()

    if arr.dtype.kind in {"S", "U"}:
        if arr.size == 1:
            return str(arr.item()).strip()

        return "".join(str(x) for x in np.ravel(arr)).strip()

    if arr.dtype.kind == "O":
        values = [
            decode_matlab_value(file, item)
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
        return str(arr.item()).strip()

    return str(arr).strip()


def get_epoch_dataset_value(dataset, trial_index, n_trials):
    if dataset.ndim == 1:
        return dataset[trial_index]

    if dataset.ndim == 2:
        if dataset.shape[0] == n_trials:
            return dataset[trial_index, 0]

        if dataset.shape[1] == n_trials:
            return dataset[0, trial_index]

    return dataset[trial_index]


def extract_epoch_text_field(file, epoch_group, field_name, n_trials):
    if field_name not in epoch_group:
        return None

    dataset = epoch_group[field_name]
    values = []

    for trial_index in range(n_trials):
        raw_value = get_epoch_dataset_value(
            dataset=dataset,
            trial_index=trial_index,
            n_trials=n_trials,
        )

        values.append(
            decode_matlab_value(
                file=file,
                value=raw_value,
            )
        )

    return values


def choose_epoch_text_field(file, epoch_group, n_trials, candidate_names):
    for field_name in candidate_names:
        values = extract_epoch_text_field(
            file=file,
            epoch_group=epoch_group,
            field_name=field_name,
            n_trials=n_trials,
        )

        if values is None:
            continue

        values = [
            str(value).strip()
            for value in values
        ]

        if any(values):
            return field_name, values

    return None, None


def extract_epoch_stimulus_table(file, condition_group, n_trials):
    if "epoch" not in condition_group:
        raise KeyError("No epoch structure found in ERP condition.")

    epoch_group = condition_group["epoch"]

    stim_file_field, stim_file_values = choose_epoch_text_field(
        file=file,
        epoch_group=epoch_group,
        n_trials=n_trials,
        candidate_names=[
            "eventstim_file",
            "stim_file",
            "eventfilename",
            "filename",
            "eventfile",
            "file",
        ],
    )

    stim_key_field, stim_key_values = choose_epoch_text_field(
        file=file,
        epoch_group=epoch_group,
        n_trials=n_trials,
        candidate_names=[
            "eventstim_key",
            "stim_key",
            "eventsentence_id",
            "sentence_id",
            "eventitem",
            "item",
        ],
    )

    trial_type_field, trial_type_values = choose_epoch_text_field(
        file=file,
        epoch_group=epoch_group,
        n_trials=n_trials,
        candidate_names=[
            "eventtrial_type",
            "trial_type",
            "eventtype",
            "type",
        ],
    )

    if stim_file_values is None and stim_key_values is None:
        raise KeyError(
            "No stim_file or stim_key field was found inside the ERP MAT epoch structure. "
            f"Available epoch fields: {list(epoch_group.keys())}"
        )

    table = pd.DataFrame(
        {
            "trial_type": trial_type_values if trial_type_values is not None else [""] * n_trials,
            "stim_file": stim_file_values if stim_file_values is not None else [""] * n_trials,
            "stim_key": stim_key_values if stim_key_values is not None else [""] * n_trials,
        }
    )

    table["trial_type"] = table["trial_type"].astype("string").str.strip().str.upper()
    table["stim_file"] = normalise_stim_file(table["stim_file"])
    table["stim_key"] = normalise_stim_key(table["stim_key"])

    print(
        f"  MAT stimulus fields: trial_type={trial_type_field}, stim_file={stim_file_field}, stim_key={stim_key_field}"
    )

    return table

def append_tsv(dataframe: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(
        path,
        sep="\t",
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )


def append_export_log(log_path: Path, row: dict):
    append_tsv(pd.DataFrame([row]), log_path)


def load_completed_export_keys(log_path: Path) -> set[tuple[str, str, str]]:
    if not log_path.exists():
        return set()

    completed = pd.read_csv(log_path, sep="\t", dtype=str)

    if completed.empty:
        return set()

    required_columns = {"subject", "analysis", "source_file", "status"}

    if not required_columns.issubset(set(completed.columns)):
        return set()

    completed = completed[completed["status"].eq("done")].copy()

    return set(
        zip(
            completed["subject"].astype(str),
            completed["analysis"].astype(str),
            completed["source_file"].astype(str),
        )
    )


def export_lookup_for_file(
    mat_path: Path,
    stimulus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    subject = get_subject_id(mat_path)
    analysis = get_analysis_name(mat_path)

    rejection_summary = load_trial_rejection_summary(mat_path)
    rejection_path = get_trial_rejection_path(mat_path)

    print(f"\nProcessing {mat_path.name}")
    print(f"  Using {rejection_path.name}")

    all_trial_lookups = []

    with h5py.File(mat_path, "r") as file:
        if "ERPs" not in file:
            raise KeyError("No ERPs dataset found in MAT file.")

        erps = file["ERPs"]

        if erps.ndim != 2:
            raise ValueError(
                "ERPs dataset has an unexpected shape: "
                f"{erps.shape}"
            )

        n_conditions = erps.shape[1]

        if len(rejection_summary) != n_conditions:
            raise ValueError(
                "Trial-rejection row count does not match "
                "the number of ERP conditions: "
                f"{len(rejection_summary)} versus {n_conditions}"
            )

        for condition_index in range(n_conditions):
            condition_number = condition_index + 1
            rejection_row = rejection_summary.iloc[condition_index]
            condition_label = str(rejection_row["condition"]).strip()
            before_trials = int(rejection_row["before_trial_rejection"])
            reported_retained_trials = int(rejection_row["after_trial_rejection"])

            condition_group = extract_condition_object(
                file=file,
                erps_dataset=erps,
                condition_index=condition_index,
            )

            data, n_trials, n_timepoints, n_channels = extract_data(
                condition_group
            )

            mat_retained_trials = int(n_trials)

            if mat_retained_trials != reported_retained_trials:
                print(
                    "  WARNING: retained-trial count mismatch for "
                    f"condition {condition_number} ({condition_label}). "
                    f"MAT file contains {mat_retained_trials}; "
                    f"{rejection_path.name} reports {reported_retained_trials}. "
                    "Using MAT-file count."
                )

            urevent_indices = extract_epoch_urevent_indices(
                file=file,
                condition_group=condition_group,
                n_trials=mat_retained_trials,
            )

            trial_lookup = pd.DataFrame(
                {
                    "subject": subject,
                    "analysis": analysis,
                    "source_file": mat_path.name,
                    "source_path": str(mat_path),
                    "condition": condition_number,
                    "condition_label": condition_label,
                    "before_trial_rejection": before_trials,
                    "after_trial_rejection_reported": reported_retained_trials,
                    "after_trial_rejection_mat": mat_retained_trials,
                    "retained_trial": np.arange(
                        1,
                        mat_retained_trials + 1,
                        dtype=int,
                    ),
                    "urevent_index": urevent_indices,
                    "original_event_row": urevent_indices,
                }
            )

            trial_lookup = trial_lookup.merge(
                stimulus_lookup[
                    [
                        "original_event_row",
                        "trial_type",
                        "stim_file",
                        "stim_key",
                    ]
                ],
                on="original_event_row",
                how="left",
                validate="many_to_one",
                sort=False,
                indicator=True,
            )

            unmatched = trial_lookup["_merge"] != "both"

            if unmatched.any():
                examples = (
                    trial_lookup.loc[
                        unmatched,
                        "original_event_row",
                    ]
                    .drop_duplicates()
                    .head(20)
                    .tolist()
                )

                raise ValueError(
                    f"{int(unmatched.sum())} retained ERP trials could not be matched to ALL_language_metrics.tsv using original_event_row/eventurevent. Examples: {examples}"
                )

            trial_lookup = trial_lookup.drop(
                columns=[
                    "_merge",
                ]
            )

            missing_stim_key = (
                trial_lookup["stim_key"].isna()
                | trial_lookup["stim_key"].astype("string").str.strip().eq("")
            )

            missing_stim_file = (
                trial_lookup["stim_file"].isna()
                | trial_lookup["stim_file"].astype("string").str.strip().eq("")
            )

            missing_trial_type = (
                trial_lookup["trial_type"].isna()
                | trial_lookup["trial_type"].astype("string").str.strip().eq("")
            )

            if missing_stim_key.any():
                raise ValueError("Some retained ERP trials have missing stim_key.")

            if missing_stim_file.any():
                raise ValueError("Some retained ERP trials have missing stim_file.")

            if missing_trial_type.any():
                raise ValueError("Some retained ERP trials have missing trial_type.")

            trial_lookup["epoch_id"] = (
                trial_lookup["subject"].astype(str)
                + "_"
                + trial_lookup["analysis"].astype(str)
                + "_c"
                + trial_lookup["condition"].astype(str)
                + "_r"
                + trial_lookup["retained_trial"].astype(str)
            )

            print(
                f"  Condition {condition_number}: "
                f"{condition_label} - "
                f"{before_trials} before rejection, "
                f"{reported_retained_trials} reported retained, "
                f"{mat_retained_trials} MAT retained"
            )

            all_trial_lookups.append(trial_lookup)

    if not all_trial_lookups:
        raise ValueError(
            "No retained-trial lookup rows were created for "
            f"{mat_path.name}."
        )

    trial_lookup_all = pd.concat(
        all_trial_lookups,
        ignore_index=True,
    )

    duplicated_lookup_rows = trial_lookup_all.duplicated(
        subset=[
            "subject",
            "analysis",
            "condition",
            "retained_trial",
        ],
        keep=False,
    )

    if duplicated_lookup_rows.any():
        examples = (
            trial_lookup_all.loc[
                duplicated_lookup_rows,
                [
                    "subject",
                    "analysis",
                    "condition",
                    "retained_trial",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"Duplicate retained-trial lookup rows were found. Examples: {examples}"
        )

    duplicated_epoch_ids = trial_lookup_all.duplicated(
        subset=[
            "epoch_id",
        ],
        keep=False,
    )

    if duplicated_epoch_ids.any():
        raise ValueError(
            f"Duplicate epoch_id values were created for {mat_path.name}."
        )

    ordered_columns = [
        "subject",
        "analysis",
        "source_file",
        "source_path",
        "condition",
        "condition_label",
        "before_trial_rejection",
        "after_trial_rejection_reported",
        "after_trial_rejection_mat",
        "retained_trial",
        "urevent_index",
        "original_event_row",
        "trial_type",
        "stim_file",
        "stim_key",
        "epoch_id",
    ]

    remaining_columns = [
        column
        for column in trial_lookup_all.columns
        if column not in ordered_columns
    ]

    return trial_lookup_all[
        ordered_columns
        + remaining_columns
    ]


def discover_erp_mat_files(
    erp_root: Path,
    analyses: list[str],
) -> list[Path]:
    """
    Find ERP MAT files for the requested analyses.

    """

    mat_files = []

    for analysis in analyses:
        nested_pattern = (
            f"sub-*/*_erp-{analysis}.mat"
        )

        direct_pattern = (
            f"*_erp-{analysis}.mat"
        )

        mat_files.extend(
            erp_root.glob(
                nested_pattern
            )
        )

        mat_files.extend(
            erp_root.glob(
                direct_pattern
            )
        )

    return sorted(
        set(
            mat_files
        )
    )


def parse_analysis_list(
    value: str,
) -> list[str]:
    """
    Parse the --analyses argument.

    Valid values:

        CP
        GA
        LD
        Order
        Time
        ALL

    Multiple analyses may be comma-separated.
    """

    valid_analyses = [
        "CP",
        "GA",
        "LD",
        "Order",
        "Time",
    ]

    cleaned_value = str(
        value
    ).strip()

    if cleaned_value.upper() == "ALL":
        return valid_analyses

    requested_analyses = [
        item.strip()
        for item in cleaned_value.split(
            ","
        )
        if item.strip()
    ]

    if not requested_analyses:
        raise ValueError(
            "No ERP analyses were requested."
        )

    normalised_lookup = {
        analysis.lower(): analysis
        for analysis in valid_analyses
    }

    normalised_analyses = []
    invalid_analyses = []

    for requested_analysis in requested_analyses:
        matched_analysis = normalised_lookup.get(
            requested_analysis.lower()
        )

        if matched_analysis is None:
            invalid_analyses.append(
                requested_analysis
            )
        else:
            normalised_analyses.append(
                matched_analysis
            )

    if invalid_analyses:
        raise ValueError(
            f"Unknown ERP analyses: {invalid_analyses}. "
            f"Valid values are {valid_analyses} or ALL."
        )

    return list(
        dict.fromkeys(
            normalised_analyses
        )
    )

def parse_subject_list(value: str) -> list[str] | None:
    cleaned_value = str(value).strip()

    if cleaned_value.upper() == "ALL":
        return None

    subjects = [
        item.strip()
        for item in cleaned_value.split(",")
        if item.strip()
    ]

    if not subjects:
        raise ValueError(
            "No subjects were requested. Use --subjects ALL "
            "or a comma-separated list such as sub-01,sub-02."
        )

    normalised_subjects = []

    for subject in subjects:
        subject = str(subject).strip()

        if subject.isdigit():
            subject = f"sub-{int(subject):02d}"

        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"

        normalised_subjects.append(subject)

    return list(dict.fromkeys(normalised_subjects))

def parse_channel_list(
    value: str,
):
    """
    Parse --channels.

    Use ALL to export every channel.

    """

    cleaned_value = str(
        value
    ).strip()

    if cleaned_value.upper() == "ALL":
        return None

    channels = {
        item.strip()
        for item in cleaned_value.split(
            ","
        )
        if item.strip()
    }

    if not channels:
        raise ValueError(
            "No channels were requested. Use --channels ALL "
            "or a comma-separated list."
        )

    return channels

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export one combined retained-trial lookup TSV "
            "from N400Stimset ERP MAT files."
        )
    )

    parser.add_argument(
        "erp_root",
        help=(
            "ERP root folder containing sub-* subject "
            "subfolders, or one subject folder."
        ),
    )

    parser.add_argument(
        "--analyses",
        default="ALL",
        help=(
            "Comma-separated ERP analyses: "
            "CP,GA,LD,Order,Time, or ALL. "
            "Default: ALL."
        ),
    )

    parser.add_argument(
        "--subjects",
        default="ALL",
        help=(
            "Comma-separated subjects such as sub-01,sub-02, "
            "or ALL. Numeric values such as 1,2 are converted "
            "to sub-01,sub-02. Default: ALL."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="eeg_outputs",
        help="Directory where the combined lookup TSV will be saved.",
    )

    parser.add_argument(
        "--language-metrics",
        required=True,
        help=(
            "Path to ALL_language_metrics.tsv containing "
            "stim_file and stim_key."
        ),
    )

    parser.add_argument(
        "--combined-name",
        default="ALL_subjects_ALL_erp_trial_lookup.tsv",
        help="Filename for the combined retained-trial lookup TSV.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip subject-analysis files already marked done "
            "in the export log."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing combined lookup and log before running.",
    )

    args = parser.parse_args()

    erp_root = Path(args.erp_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    language_metrics_path = Path(args.language_metrics).expanduser().resolve()
    analyses = parse_analysis_list(args.analyses)
    subjects = parse_subject_list(args.subjects)

    output_dir.mkdir(parents=True, exist_ok=True)

    combined_path = output_dir / args.combined_name
    log_path = output_dir / "ALL_subjects_ALL_erp_trial_lookup_export_log.tsv"
    failed_path = output_dir / "failed_erp_lookup_exports.tsv"

    if args.overwrite:
        for path in [
            combined_path,
            log_path,
            failed_path,
        ]:
            if path.exists():
                path.unlink()

    if combined_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            "Combined lookup already exists. Use --resume or --overwrite: "
            f"{combined_path}"
        )

    if not erp_root.exists():
        raise FileNotFoundError(
            f"ERP root not found: {erp_root}"
        )

    if not erp_root.is_dir():
        raise NotADirectoryError(
            "ERP root is not a directory: "
            f"{erp_root}"
        )

    stimulus_lookup = load_stimulus_lookup(
        language_metrics_path
    )

    mat_files = discover_erp_mat_files(
        erp_root=erp_root,
        analyses=analyses,
    )

    if subjects is not None:
        requested_subjects = set(
            subjects
        )

        mat_files = [
            mat_path
            for mat_path in mat_files
            if get_subject_id(
                mat_path
            )
            in requested_subjects
        ]

    if not mat_files:
        searched_patterns = "\n".join(
            [
                str(
                    erp_root
                    / "sub-*"
                    / f"*_erp-{analysis}.mat"
                )
                for analysis in analyses
            ]
            + [
                str(
                    erp_root
                    / f"*_erp-{analysis}.mat"
                )
                for analysis in analyses
            ]
        )

        if subjects is None:
            subject_message = "Subjects: ALL"
        else:
            subject_message = "Subjects: " + ", ".join(
                subjects
            )

        raise FileNotFoundError(
            "No matching ERP MAT files were found. "
            f"{subject_message}\n"
            "Searched:\n"
            f"{searched_patterns}"
        )

    if subjects is not None:
        found_subjects = {
            get_subject_id(
                mat_path
            )
            for mat_path in mat_files
        }

        missing_subjects = [
            subject
            for subject in subjects
            if subject not in found_subjects
        ]

        if missing_subjects:
            raise FileNotFoundError(
                "No ERP MAT files were found for requested subjects: "
                f"{missing_subjects}"
            )

    missing_rejection_files = [
        get_trial_rejection_path(
            mat_path
        )
        for mat_path in mat_files
        if not get_trial_rejection_path(
            mat_path
        ).exists()
    ]

    if missing_rejection_files:
        examples = "\n".join(
            str(path)
            for path in missing_rejection_files[:20]
        )

        raise FileNotFoundError(
            "Some ERP MAT files do not have their exact "
            "matching *_trialrej.tsv file:\n"
            f"{examples}"
        )

    completed_keys = (
        load_completed_export_keys(
            log_path
        )
        if args.resume
        else set()
    )

    print(
        f"Found {len(mat_files)} ERP MAT files."
    )

    print(
        f"Using ERP root: {erp_root}"
    )

    print(
        "Analyses: " + ", ".join(
            analyses
        )
    )

    if subjects is None:
        print(
            "Subjects: ALL"
        )
    else:
        print(
            "Subjects: " + ", ".join(
                subjects
            )
        )

    print(
        f"Language metrics: {language_metrics_path}"
    )

    print(
        f"Combined lookup: {combined_path}"
    )

    success_count = 0
    skipped_count = 0
    failed_count = 0

    for mat_path in mat_files:
        subject = get_subject_id(
            mat_path
        )

        analysis = get_analysis_name(
            mat_path
        )

        source_file = mat_path.name

        key = (
            subject,
            analysis,
            source_file,
        )

        if key in completed_keys:
            print(
                f"SKIP already done: {source_file}"
            )

            skipped_count += 1

            continue

        try:
            lookup = export_lookup_for_file(
                mat_path=mat_path,
                stimulus_lookup=stimulus_lookup,
            )

            append_tsv(
                lookup,
                combined_path,
            )

            append_export_log(
                log_path,
                {
                    "subject": subject,
                    "analysis": analysis,
                    "source_file": source_file,
                    "source_path": str(
                        mat_path
                    ),
                    "status": "done",
                    "n_rows": len(
                        lookup
                    ),
                    "error": "",
                },
            )

            success_count += 1

            print(
                f"Saved {len(lookup)} rows from {source_file}"
            )

            del lookup

        except Exception as error:
            failed_count += 1

            error_message = str(
                error
            )

            print(
                f"FAILED: {source_file}: {error_message}"
            )

            failed_row = {
                "subject": subject,
                "analysis": analysis,
                "source_file": source_file,
                "source_path": str(
                    mat_path
                ),
                "status": "failed",
                "n_rows": 0,
                "error": error_message,
            }

            append_export_log(
                log_path,
                failed_row,
            )

            append_tsv(
                pd.DataFrame(
                    [
                        failed_row
                    ]
                ),
                failed_path,
            )

    if not combined_path.exists():
        raise RuntimeError(
            "No lookup rows were exported successfully."
        )

    combined_check = pd.read_csv(
        combined_path,
        sep="\t",
        usecols=[
            "subject",
        ],
    )

    if combined_check.empty:
        raise RuntimeError(
            "Combined lookup file exists but is empty."
        )

    print(
        f"Successful exports: {success_count}"
    )

    print(
        f"Skipped exports: {skipped_count}"
    )

    print(
        f"Failed exports: {failed_count}"
    )

    print(
        f"Final combined lookup: {combined_path}"
    )

    print(
        f"Export log: {log_path}"
    )

    if failed_count:
        print(
            f"Failed export report: {failed_path}"
        )

    print(
        "Done."
    )

if __name__ == "__main__":
    main()