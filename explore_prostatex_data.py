from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import pydicom
import dcm_classifier
from dcm_classifier.image_type_inference import ImageTypeClassifierBase
from dcm_classifier.study_processing import ProcessOneDicomStudyToVolumesMappingBase

import botimageai


def parse_subject_data(subject_path: Path):
    subject_data = {}
    subject_data["subject_path"] = subject_path
    for dir in subject_path.iterdir():
        print(f"dir: {dir}")
        print(f"dir.name: {dir.name}")
        children = list(dir.glob("*"))
        print(f"dir.glob('*'): {[x.name for x in children]}")
        for child in children:
            print(f"child: {child}")
            print(f"child.name: {child.name}")
            dcm_file = list(child.glob("*.dcm"))[0]
            dcm_data = pydicom.dcmread(dcm_file, stop_before_pixels=True)
            if "MR" in dcm_data.Modality:
                pass


## dcm_classifier.py scripts
def get_model_path() -> Path:
    return (
        botimageai.__get_botimage_resources_root__()
        / "dicom_processing/ova_rf_classifier_20240314.onnx"
    )


def generate_separator(column_width):
    return "|" + ("-" * (column_width + 2) + "|") * 4


def generate_row(*args, column_width):
    return "| " + " | ".join(arg.ljust(column_width) for arg in args) + " |"


def do_single_session_classification_guess(session_directory: Path):
    inferer = ImageTypeClassifierBase(classification_model_filename=get_model_path())
    study = ProcessOneDicomStudyToVolumesMappingBase(
        study_directory=session_directory, inferer=inferer
    )
    study.run_inference()

    list_of_inputs: list[dict[str, Any]] = []
    list_of_probabilities: list[pd.DataFrame] = []
    list_of_dictionaries: list[dict[str, str]] = []

    for series_number, series in study.series_dictionary.items():
        for index, volume in enumerate(series.get_volume_list()):
            # modality = volume.get_modality()
            #         plane = volume.get_acquisition_plane()
            #         print(generate_row(str(series_number), modality, plane, iso, column_width=col_width))
            current_dict: dict[str, str] = {}
            current_dict["Series#"] = str(series_number)
            current_dict["Vol.#"] = str(volume.get_volume_index())
            current_dict["Volume Modality"] = str(volume.get_volume_modality())
            current_dict["Series Modality"] = str(series.get_series_modality())
            current_dict["Acq.Plane"] = str(volume.get_acquisition_plane())
            current_dict["Isotropic"] = str(volume.get_is_isotropic())
            print(volume.get_modality_probabilities().to_string(index=False))
            current_dict["Bvalue"] = str(volume.get_volume_bvalue())
            # info_dict = series.get_series_info_dict()
            inputs_df: dict[str, Any] = volume.get_volume_dictionary()
            for unwanted in [
                "FileName",
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "list_of_ordered_volume_files",
            ]:
                if unwanted in inputs_df:
                    inputs_df.pop(unwanted)

            list_of_inputs.append(inputs_df)

            prob_df: pd.DataFrame = volume.get_modality_probabilities()
            list_of_probabilities.append(prob_df)
            list_of_dictionaries.append(current_dict)

    df: pd.DataFrame = pd.DataFrame(list_of_dictionaries)
    return df


def return_matched_segmentation_and_mr_data(
    mr_data_path: Path, segmentation_data_path: Path
):
    matching_dir = {}
    for dir in mr_data_path.iterdir():
        if "ProstateX" in dir.name:
            matched_seg_path = segmentation_data_path / dir.name
            if matched_seg_path.exists():
                matching_dir[dir.name] = {
                    "mr_data": dir,
                    "segmentation_data": matched_seg_path,
                }

    return matching_dir


def create_purged_subject_dir(subject_path: Path, output_dir: Path):
    modalities_to_purge = ["t1w", "gret2star"]
    pass


if __name__ == "__main__":
    # manifest-1605042674814 contains only segmentation data

    base_data_dir = Path(
        "/Users/iejohnson/School/spring_2024/AML/Supervised_learning/ProstateX/ProstateX_DICOM"
    )
    segmentation_data_dir = base_data_dir / "manifest-1605042674814" / "PROSTATEx"
    subjects_dir = base_data_dir / "manifest-A3Y4AE4o5818678569166032044" / "PROSTATEx"
    df = pd.DataFrame()
    matching_dir = return_matched_segmentation_and_mr_data(
        subjects_dir, segmentation_data_dir
    )
    print(f"Found {len(matching_dir)} matching directories")
    subjects_without_segmentations = [
        x for x in subjects_dir.iterdir() if x.name not in matching_dir.keys()
    ]
    print(f"Found {len(subjects_without_segmentations)} subjects without segmentations")

    for subject_path in subjects_without_segmentations[:5]:
        if "ProstateX" in subject_path.name:
            print(f"*" * 80)
            print(f"subject_path: {subject_path}")
            subject_df = do_single_session_classification_guess(subject_path)
            print(f"*" * 80)
            print(subject_df)
