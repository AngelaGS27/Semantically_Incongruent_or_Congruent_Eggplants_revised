"""
Prepare a subject-specific design matrix for LIMO.

The script:

1. Loads the subject events.tsv.
2. Extracts a stable stimulus identifier as stim_key.
3. Optionally keeps only trials listed in a surviving-trials file.
4. Matches each trial to the language predictors using stim_key.
5. Validates duplicates, unmatched trials, and trial order.
6. Saves a subject-specific design matrix.

This script prepares the input table used by the LIMO analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


STIMULUS_COLUMNS = [
    "stim_key",
    "stim_file",
    "stimulus",
    "sentence_id",
    "item",
    "trial",
]


def normalise_stim_key(series: pd.Series) -> pd.Series:
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

def load_stimulus_lookup(
    language_metrics_path: Path,
) -> pd.DataFrame:
    """
    Load the complete language metrics table and create a validated
    stim_file -> stim_key lookup.

    Expected input:
        language_outputs/ALL_language_metrics.tsv

    This function is retained for workflows that start directly from
    events.tsv. The main retained-trial workflow normally receives
    stim_file and stim_key from export_erp_long.py.
    """

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

    stimuli = pd.read_csv(
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
        if column not in stimuli.columns
    ]

    if missing_columns:
        raise ValueError(
            "Language metrics table is missing required "
            f"columns: {missing_columns}. "
            "Use ALL_language_metrics.tsv, not "
            "ALL_language_metrics_GLM.tsv, for this lookup."
        )

    stimuli = stimuli.copy()

    stimuli["stim_file"] = normalise_stim_file(
        stimuli["stim_file"]
    )

    stimuli["stim_key"] = normalise_stim_key(
        stimuli["stim_key"]
    )

    missing_stim_files = (
        stimuli["stim_file"].isna()
        | stimuli["stim_file"].eq("")
    )

    if missing_stim_files.any():
        raise ValueError(
            "Language metrics table contains "
            f"{int(missing_stim_files.sum())} missing or empty "
            "stim_file values."
        )

    missing_stim_keys = (
        stimuli["stim_key"].isna()
        | stimuli["stim_key"].eq("")
    )

    if missing_stim_keys.any():
        raise ValueError(
            "Language metrics table contains "
            f"{int(missing_stim_keys.sum())} missing or empty "
            "stim_key values."
        )

    conflicting = (
        stimuli
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

    lookup = (
        stimuli[
            [
                "stim_file",
                "stim_key",
            ]
        ]
        .drop_duplicates(
            subset="stim_file",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Loaded {len(lookup)} stimulus mappings "
        f"from {language_metrics_path}"
    )

    return lookup

def load_language_predictors(predictors_path: Path) -> pd.DataFrame:
    """
    Load the z-scored language predictor table.

    Expected input:
        ALL_language_metrics_GLM.tsv
    """
    predictors = pd.read_csv(
        predictors_path,
        sep=None,
        engine="python",
    )

    if "stim_key" not in predictors.columns:
        raise ValueError(
            f"Predictor table does not contain 'stim_key': {predictors_path}"
        )

    predictors["stim_key"] = normalise_stim_key(
        predictors["stim_key"]
    )

    if predictors["stim_key"].isna().any():
        n_missing = predictors["stim_key"].isna().sum()

        raise ValueError(
            f"Predictor table contains {n_missing} missing stim_key values."
        )

    duplicated = predictors["stim_key"].duplicated(
        keep=False
    )

    if duplicated.any():
        examples = (
            predictors.loc[duplicated, "stim_key"]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Predictor table contains duplicate stim_key values. "
            f"Examples: {examples}"
        )

    return predictors

def normalise_subject_id(value: str) -> str:
    value = str(value).strip()

    if value.isdigit():
        return f"sub-{int(value):02d}"

    if not value.startswith("sub-"):
        return f"sub-{value}"

    return value


def parse_subject_list(value: str) -> list[str]:
    subjects = [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]

    if not subjects:
        raise ValueError(
            "No subjects were requested."
        )

    return list(
        dict.fromkeys(
            normalise_subject_id(subject)
            for subject in subjects
        )
    )

def load_trial_lookup(
    trial_lookup_path: Path,
    subject: str,
    analysis: str,
) -> pd.DataFrame:
    trial_lookup_path = Path(
        trial_lookup_path
    ).expanduser().resolve()

    if not trial_lookup_path.exists():
        raise FileNotFoundError(
            "Trial lookup file not found: "
            f"{trial_lookup_path}"
        )

    lookup = pd.read_csv(
        trial_lookup_path,
        sep="\t",
    )

    required_columns = [
        "subject",
        "analysis",
        "condition",
        "condition_label",
        "retained_trial",
        "urevent_index",
        "original_event_row",
        "trial_type",
        "stim_file",
        "stim_key",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in lookup.columns
    ]

    if missing_columns:
        raise ValueError(
            "Trial lookup is missing required columns: "
            f"{missing_columns}"
        )

    if lookup.empty:
        raise ValueError(
            f"Trial lookup is empty: {trial_lookup_path}"
        )

    lookup = lookup.copy()

    lookup["subject"] = (
        lookup["subject"]
        .astype("string")
        .str.strip()
    )

    lookup["analysis"] = (
        lookup["analysis"]
        .astype("string")
        .str.strip()
    )

    subject = normalise_subject_id(
        subject
    )

    analysis = str(
        analysis
    ).strip()

    lookup = lookup[
        lookup["subject"].eq(subject)
        & lookup["analysis"].eq(analysis)
    ].copy()

    if lookup.empty:
        raise ValueError(
            "No retained-trial lookup rows found for "
            f"subject={subject}, analysis={analysis}."
        )

    lookup["stim_key"] = normalise_stim_key(
        lookup["stim_key"]
    )

    lookup["stim_file"] = normalise_stim_file(
        lookup["stim_file"]
    )

    integer_columns = [
        "condition",
        "retained_trial",
        "urevent_index",
        "original_event_row",
    ]

    for column in integer_columns:
        lookup[column] = pd.to_numeric(
            lookup[column],
            errors="raise",
        ).astype(int)

    missing_stim_keys = (
        lookup["stim_key"].isna()
        | lookup["stim_key"].eq("")
    )

    if missing_stim_keys.any():
        raise ValueError(
            "Filtered trial lookup contains "
            f"{int(missing_stim_keys.sum())} missing or empty "
            "stim_key values."
        )

    missing_stim_files = (
        lookup["stim_file"].isna()
        | lookup["stim_file"].eq("")
    )

    if missing_stim_files.any():
        raise ValueError(
            "Filtered trial lookup contains "
            f"{int(missing_stim_files.sum())} missing or empty "
            "stim_file values."
        )

    duplicated_retained_trials = lookup.duplicated(
        subset=[
            "condition",
            "retained_trial",
        ],
        keep=False,
    )

    if duplicated_retained_trials.any():
        examples = (
            lookup.loc[
                duplicated_retained_trials,
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
            "Filtered trial lookup contains duplicate "
            "condition/retained_trial combinations. "
            f"Examples: {examples}"
        )

    duplicated_events = lookup.duplicated(
        subset=[
            "original_event_row",
        ],
        keep=False,
    )

    if duplicated_events.any():
        examples = (
            lookup.loc[
                duplicated_events,
                [
                    "subject",
                    "analysis",
                    "condition",
                    "retained_trial",
                    "original_event_row",
                    "stim_key",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "The same original event is assigned to more than "
            "one retained ERP trial after filtering. "
            f"Examples: {examples}"
        )

    lookup = lookup.reset_index(
        drop=True
    )

    lookup["eeg_trial"] = range(
        1,
        len(lookup) + 1,
    )

    return lookup

def find_stimulus_column(events: pd.DataFrame) -> str:
    stim_col = next(
        (
            col
            for col in STIMULUS_COLUMNS
            if col in events.columns
        ),
        None,
    )

    if stim_col is None:
        raise ValueError(
            "No stimulus identifier was found in events.tsv. "
            f"Checked: {STIMULUS_COLUMNS}"
        )

    return stim_col


def load_subject_events(
    events_path: Path,
    stimulus_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load one subject's events.tsv.

    Only NPC and NPI sentence trials are retained. stim_key is
    attached by matching stim_file to the lookup constructed from
    language_outputs/ALL_language_metrics.tsv.

    This function is retained for workflows that operate directly
    from events.tsv. The current main workflow uses the retained-trial
    lookup created by export_erp_long.py.
    """

    events_path = Path(
        events_path
    ).expanduser().resolve()

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}"
        )

    events = pd.read_csv(
        events_path,
        sep="\t",
    )

    if events.empty:
        raise ValueError(
            f"Events table is empty: {events_path}"
        )

    required_columns = [
        "trial_type",
        "stim_file",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in events.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{events_path} is missing required "
            f"columns: {missing_columns}"
        )

    events = events.copy()

    # Preserve the row in the complete events table before filtering.
    events["original_event_row"] = range(
        1,
        len(events) + 1,
    )

    events["trial_type"] = (
        events["trial_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    events = events[
        events["trial_type"].isin(
            [
                "NPC",
                "NPI",
            ]
        )
    ].copy()

    if events.empty:
        raise ValueError(
            f"No NPC or NPI sentence trials found in {events_path}"
        )

    events["stim_file"] = normalise_stim_file(
        events["stim_file"]
    )

    missing_stim_files = (
        events["stim_file"].isna()
        | events["stim_file"].eq("")
    )

    if missing_stim_files.any():
        raise ValueError(
            f"{int(missing_stim_files.sum())} NPC/NPI events "
            "contain missing or empty stim_file values."
        )

    events = events.merge(
        stimulus_lookup,
        on="stim_file",
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    unmatched = (
        events["_merge"] != "both"
    )

    if unmatched.any():
        examples = (
            events.loc[
                unmatched,
                "stim_file",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"{int(unmatched.sum())} events could not be matched "
            "to ALL_language_metrics.tsv using stim_file. "
            f"Examples: {examples}"
        )

    events = events.drop(
        columns="_merge"
    )

    events["subject_trial"] = range(
        1,
        len(events) + 1,
    )

    condition_map = {
        "NPC": 1,
        "NPI": 2,
    }

    events["condition"] = (
        events["trial_type"]
        .map(
            condition_map
        )
        .astype(int)
    )

    events["condition_trial"] = (
        events
        .groupby(
            "condition",
            sort=False,
        )
        .cumcount()
        + 1
    )

    print(
        f"Loaded {len(events)} experimental trials: "
        f"{int((events['trial_type'] == 'NPC').sum())} NPC and "
        f"{int((events['trial_type'] == 'NPI').sum())} NPI"
    )

    return events.reset_index(
        drop=True
    )


def load_surviving_trials(
    surviving_trials_path: Path,
) -> pd.DataFrame:
    """
    Load a table specifying which trials remain in the cleaned EEG data.

    Accepted trial-index columns:
        original_event_row
        subject_trial
        trial
        epoch
        epoch_index
    """
    surviving = pd.read_csv(
        surviving_trials_path,
        sep=None,
        engine="python",
    )

    candidate_columns = [
        "original_event_row",
        "subject_trial",
        "trial",
        "epoch",
        "epoch_index",
    ]

    trial_col = next(
        (
            col
            for col in candidate_columns
            if col in surviving.columns
        ),
        None,
    )

    if trial_col is None:
        raise ValueError(
            "Surviving-trials table does not contain a recognised "
            f"trial-index column. Checked: {candidate_columns}"
        )

    surviving = surviving.copy()

    surviving["original_event_row"] = pd.to_numeric(
        surviving[trial_col],
        errors="raise",
    ).astype(int)

    if surviving["original_event_row"].duplicated().any():
        raise ValueError(
            "Surviving-trials table contains duplicate trial indices."
        )

    return surviving[["original_event_row"]]


def restrict_to_surviving_trials(
    events: pd.DataFrame,
    surviving_trials: pd.DataFrame,
) -> pd.DataFrame:
    """
    Restrict events to trials that remain in the cleaned EEG file.

    The order of surviving_trials is preserved.
    """
    surviving_trials = surviving_trials.copy()

    surviving_trials["eeg_trial"] = range(
        1,
        len(surviving_trials) + 1,
    )

    filtered = surviving_trials.merge(
        events,
        on="original_event_row",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if filtered["stim_key"].isna().any():
        missing_rows = filtered.loc[
            filtered["stim_key"].isna(),
            "original_event_row",
        ].tolist()

        raise ValueError(
            "Some surviving trial indices were not found in events.tsv: "
            f"{missing_rows[:10]}"
        )

    filtered = filtered.sort_values(
        "eeg_trial"
    ).reset_index(drop=True)

    return filtered


def build_subject_design_matrix(
    trial_lookup: pd.DataFrame,
    predictors: pd.DataFrame,
) -> pd.DataFrame:
    if "stim_key" not in trial_lookup.columns:
        raise ValueError(
            "Trial lookup does not contain stim_key."
        )

    if "stim_key" not in predictors.columns:
        raise ValueError(
            "Predictor table does not contain stim_key."
        )

    trial_lookup = trial_lookup.copy()
    predictors = predictors.copy()

    trial_lookup["stim_key"] = normalise_stim_key(
        trial_lookup["stim_key"]
    )

    predictors["stim_key"] = normalise_stim_key(
        predictors["stim_key"]
    )

    duplicated_predictors = predictors.duplicated(
        subset=[
            "stim_key",
        ],
        keep=False,
    )

    if duplicated_predictors.any():
        examples = (
            predictors.loc[
                duplicated_predictors,
                "stim_key",
            ]
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Predictor table contains duplicate stim_key values. "
            f"Examples: {examples}"
        )

    design = trial_lookup.merge(
        predictors,
        on="stim_key",
        how="left",
        validate="many_to_one",
        sort=False,
        suffixes=(
            "",
            "_predictor",
        ),
        indicator=True,
    )

    unmatched = (
        design["_merge"]
        != "both"
    )

    if unmatched.any():
        examples = (
            design.loc[
                unmatched,
                [
                    "stim_key",
                    "stim_file",
                ],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            f"{int(unmatched.sum())} retained EEG trials have "
            "no matched language predictors. "
            f"Examples: {examples}"
        )

    design = design.drop(
        columns=[
            "_merge",
        ]
    )

    predictor_columns = [
        column
        for column in predictors.columns
        if column != "stim_key"
    ]

    completely_missing = (
        design[
            predictor_columns
        ]
        .isna()
        .all(axis=1)
    )

    if completely_missing.any():
        raise ValueError(
            f"{int(completely_missing.sum())} retained trials "
            "contain no predictor values."
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
        "eeg_trial",
        "urevent_index",
        "original_event_row",
        "trial_type",
        "stim_file",
        "stim_key",
        "epoch_id",
    ]

    existing_ordered_columns = [
        column
        for column in ordered_columns
        if column in design.columns
    ]

    remaining_columns = [
        column
        for column in design.columns
        if column not in existing_ordered_columns
    ]

    return design[
        existing_ordered_columns
        + remaining_columns
    ]


def validate_design_matrix(
    design: pd.DataFrame,
    predictors: pd.DataFrame,
) -> None:
    if design.empty:
        raise ValueError(
            "The final design matrix is empty."
        )

    required_columns = [
        "subject",
        "analysis",
        "condition",
        "condition_label",
        "retained_trial",
        "eeg_trial",
        "urevent_index",
        "original_event_row",
        "trial_type",
        "stim_file",
        "stim_key",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in design.columns
    ]

    if missing_columns:
        raise ValueError(
            "The final design matrix is missing required columns: "
            f"{missing_columns}"
        )

    if design["subject"].nunique(dropna=True) != 1:
        raise ValueError(
            "The final design matrix must contain exactly one subject."
        )

    if design["analysis"].nunique(dropna=True) != 1:
        raise ValueError(
            "The final design matrix must contain exactly one analysis."
        )

    design["eeg_trial"] = pd.to_numeric(
        design["eeg_trial"],
        errors="raise",
    ).astype(int)

    expected_eeg_trial = list(
        range(
            1,
            len(design) + 1,
        )
    )

    actual_eeg_trial = design[
        "eeg_trial"
    ].tolist()

    if actual_eeg_trial != expected_eeg_trial:
        raise ValueError(
            "eeg_trial must be consecutive and in design-row order."
        )

    duplicated_trials = design.duplicated(
        subset=[
            "condition",
            "retained_trial",
        ],
        keep=False,
    )

    if duplicated_trials.any():
        examples = (
            design.loc[
                duplicated_trials,
                [
                    "condition",
                    "retained_trial",
                ],
            ]
            .head(10)
            .to_dict("records")
        )

        raise ValueError(
            "The final design matrix contains duplicate "
            "condition/retained_trial combinations. "
            f"Examples: {examples}"
        )

    duplicated_eeg_trials = design["eeg_trial"].duplicated(
        keep=False
    )

    if duplicated_eeg_trials.any():
        raise ValueError(
            "The final design matrix contains duplicate eeg_trial values."
        )

    missing_stim_key = (
        design["stim_key"].isna()
        | design["stim_key"].astype("string").str.strip().eq("")
    )

    if missing_stim_key.any():
        raise ValueError(
            "The final design matrix contains missing stim_key values."
        )

    predictor_columns = [
        column
        for column in predictors.columns
        if column != "stim_key"
    ]

    non_numeric = []

    for column in predictor_columns:
        converted = pd.to_numeric(
            design[column],
            errors="coerce",
        )

        if converted.notna().sum() == 0:
            non_numeric.append(
                column
            )

    if non_numeric:
        raise ValueError(
            "These predictor columns are not numeric: "
            + ", ".join(non_numeric)
        )

    print()
    print("Design-matrix validation")
    print("------------------------")
    print(f"Subject: {design['subject'].iloc[0]}")
    print(f"Analysis: {design['analysis'].iloc[0]}")
    print(f"Rows: {len(design)}")
    print(f"Unique stimuli: {design['stim_key'].nunique()}")
    print(f"Predictor columns: {len(predictor_columns)}")
    print(
        "Rows with any missing predictor value:",
        int(
            design[
                predictor_columns
            ]
            .isna()
            .any(axis=1)
            .sum()
        ),
    )


def save_design_matrix(
    design: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Save the subject-specific design matrix.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    design.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    print(f"Saved design matrix: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build subject-specific LIMO design matrices "
            "from the combined retained ERP trial lookup and "
            "language predictors."
        )
    )

    parser.add_argument(
        "--trial-lookup",
        required=True,
        help=(
            "Path to ALL_subjects_ALL_erp_trial_lookup.tsv."
        ),
    )

    parser.add_argument(
        "--predictors",
        required=True,
        help=(
            "Path to ALL_language_metrics_GLM.tsv."
        ),
    )

    parser.add_argument(
        "--analysis",
        required=True,
        help=(
            "ERP analysis to prepare, such as CP, GA, LD, Order, or Time."
        ),
    )

    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "One subject, such as sub-01 or 1."
        ),
    )

    parser.add_argument(
        "--subjects",
        default=None,
        help=(
            "Comma-separated subjects, such as sub-01,sub-02 or 1,2."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output path for one subject design matrix TSV. "
            "Use only with --subject."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for one or more subject design matrices."
        ),
    )

    args = parser.parse_args()

    trial_lookup_path = Path(
        args.trial_lookup
    )

    predictors_path = Path(
        args.predictors
    )

    analysis = str(
        args.analysis
    ).strip()

    if args.subject is not None and args.subjects is not None:
        raise ValueError(
            "Use either --subject or --subjects, not both."
        )

    if args.subject is None and args.subjects is None:
        raise ValueError(
            "You must provide --subject or --subjects."
        )

    if args.subject is not None:
        subjects = [
            normalise_subject_id(
                args.subject
            )
        ]
    else:
        subjects = parse_subject_list(
            args.subjects
        )

    if args.output is not None and len(subjects) != 1:
        raise ValueError(
            "--output can only be used with exactly one subject."
        )

    if args.output is None and args.output_dir is None:
        raise ValueError(
            "Use --output for one subject or --output-dir for one or more subjects."
        )

    if args.output is not None and args.output_dir is not None:
        raise ValueError(
            "Use either --output or --output-dir, not both."
        )

    if not trial_lookup_path.exists():
        raise FileNotFoundError(
            "Trial lookup file not found: "
            f"{trial_lookup_path}"
        )

    if not predictors_path.exists():
        raise FileNotFoundError(
            f"Predictor file not found: {predictors_path}"
        )

    predictors = load_language_predictors(
        predictors_path
    )

    written_outputs = []

    for subject in subjects:
        trial_lookup = load_trial_lookup(
            trial_lookup_path=trial_lookup_path,
            subject=subject,
            analysis=analysis,
        )

        design = build_subject_design_matrix(
            trial_lookup=trial_lookup,
            predictors=predictors,
        )

        validate_design_matrix(
            design=design,
            predictors=predictors,
        )

        if args.output is not None:
            output_path = Path(
                args.output
            )
        else:
            output_dir = Path(
                args.output_dir
            )

            output_path = (
                output_dir
                / f"{subject}_erp-{analysis}_design_matrix.tsv"
            )

        save_design_matrix(
            design=design,
            output_path=output_path,
        )

        written_outputs.append(
            output_path
        )

    print()
    print("Design matrices written:")
    for output_path in written_outputs:
        print(output_path)


if __name__ == "__main__":
    main()