import random

import itk
from pathlib import Path

from matplotlib import pyplot as plt

from data_connector import get_exploratory_image_info, get_image_stats
import pandas as pd
import itertools


def show_subject_images(image: itk.Image, mask: itk.Image):
    # Convert the images to numpy arrays
    image_np = itk.GetArrayViewFromImage(image)
    mask_np = itk.GetArrayViewFromImage(mask)

    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np[5, :, :], cmap="gray")
    ax[0].set_title("Image")
    ax[1].imshow(mask_np[5, :, :], cmap="gray")
    ax[1].set_title("Mask")
    plt.show()


def init_data_lists(base_data_path):
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            image_paths.append(dir / f"{dir.name}_prostate.nii.gz")
            mask_paths.append(dir / f"{dir.name}_segmentation.nii.gz")
    return image_paths, mask_paths


def get_mask_bounding_box(
    mask: itk.Image, label: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    # Initialize the LabelStatisticsImageFilter
    stats_filter = itk.LabelStatisticsImageFilter.New(Input=mask, LabelInput=mask)
    stats_filter.Update()

    # Extract the bounding box for the specified label
    bounding_box = stats_filter.GetBoundingBox(label)
    # Unpack the bounding box values
    x_start, x_end, y_start, y_end, z_start, z_end = bounding_box
    print(
        f"X Start: {x_start}, X End: {x_end}, Y Start: {y_start}, Y End: {y_end}, Z Start: {z_start}, Z End: {z_end}"
    )
    return (x_start, y_start, z_start), (x_end, y_end, z_end)


def calculate_bounding_box_extent(mask_image: itk.Image) -> tuple[int, int, int]:
    # Get the bounding box
    bounding_box = get_mask_bounding_box(mask_image, 1)
    x_extent = bounding_box[1][0] - bounding_box[0][0]
    y_extent = bounding_box[1][1] - bounding_box[0][1]
    z_extent = bounding_box[1][2] - bounding_box[0][2]
    print(f"X Extent: {x_extent}, Y Extent: {y_extent}, Z Extent: {z_extent}")
    return x_extent, y_extent, z_extent


def calculate_bounding_box_extent_in_mm(mask_image):
    # Get the image spacing
    spacing = mask_image.GetSpacing()
    # Get the bounding box
    x_extent, y_extent, z_extent = calculate_bounding_box_extent(mask_image)
    x_extent_mm = x_extent * spacing[0]
    y_extent_mm = y_extent * spacing[1]
    z_extent_mm = z_extent * spacing[2]
    # Calculate the extent in mm
    extent_str = f"X Extent (mm): {x_extent_mm}, Y Extent (mm): {y_extent_mm}, Z Extent (mm): {z_extent_mm}"
    return extent_str


def convert_multiclass_mask_to_binary(mask: itk.Image) -> itk.Image:
    # Initialize the BinaryThresholdImageFilter
    binary_filter = itk.BinaryThresholdImageFilter.New(
        Input=mask, LowerThreshold=0, UpperThreshold=0, InsideValue=0, OutsideValue=1
    )
    binary_filter.Update()

    return binary_filter.GetOutput()


def get_initial_reportable_image_stats(itk_image: itk.Image) -> dict:
    mean, std, max, min = get_image_stats(itk_image)
    spacing = list(itk_image.GetSpacing())
    size = list(itk_image.GetLargestPossibleRegion().GetSize())
    return {
        "max": max,
        "mean": mean,
        "min": min,
        "std": std,
        "spacing": spacing,
        "size": size,
    }


def write_exploratry_image_info(image_paths, mask_paths, output_path):
    image_df = pd.DataFrame()
    mask_df = pd.DataFrame()
    for image_path, mask_path in zip(image_paths, mask_paths):
        print(image_path, mask_path)
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)
        image_info = get_initial_reportable_image_stats(image)
        mask_info = get_initial_reportable_image_stats(mask)
        image_info["image_name"] = image_path.name
        mask_info["mask_name"] = mask_path.name

        print(f"Image Info: {image_info}")
        print(f"Mask Info: {mask_info}")
        # show_subject_images(image, mask)
        # Convert the multiclass mask to a binary mask
        binary_mask = convert_multiclass_mask_to_binary(mask)

        # show_subject_images(image,binary_mask)

        # Calculate the extent of the bounding box
        prostate_extent = calculate_bounding_box_extent(binary_mask)
        x_extent, y_extent, z_extent = prostate_extent
        mask_info["prostate_x_extent"] = x_extent
        mask_info["prostate_y_extent"] = y_extent
        mask_info["prostate_z_extent"] = z_extent
        prostate_extent_mm = calculate_bounding_box_extent_in_mm(binary_mask)
        mask_info["prostate_extent_mm"] = prostate_extent_mm
        if image_df.empty:
            image_df = pd.Series(image_info).to_frame().T
            mask_df = pd.Series(mask_info).to_frame().T
        else:
            image_df = pd.concat(
                [image_df, pd.Series(image_info).to_frame().T], ignore_index=True
            )
            mask_df = pd.concat(
                [mask_df, pd.Series(mask_info).to_frame().T], ignore_index=True
            )
    print(image_df.head())
    image_df.to_csv("image_info.csv", index=False)
    mask_df.to_csv("mask_info.csv", index=False)


def find_center_of_gravity(image: itk.Image) -> tuple[int, int, int]:

    UC_Image_Type = itk.Image[itk.UC, 3]
    UC_image = cast_to_unsigned_char(image)

    moments_filter = itk.ImageMomentsCalculator[UC_Image_Type].New()
    moments_filter.SetImage(UC_image)
    moments_filter.Compute()
    center_of_gravity_in_physical_space = moments_filter.GetCenterOfGravity()
    center_of_gravity = image.TransformPhysicalPointToContinuousIndex(
        center_of_gravity_in_physical_space
    )

    return center_of_gravity


def cast_to_unsigned_char(image: itk.Image) -> itk.Image:
    UC_Image_Type = itk.Image[itk.UC, 3]
    cast_image_filter = itk.CastImageFilter[image, UC_Image_Type].New()
    cast_image_filter.SetInput(image)
    cast_image_filter.Update()
    return cast_image_filter.GetOutput()


def simple_otsu_thresholding(image: itk.Image) -> itk.Image:
    # Initialize the OtsuThresholdImageFilter
    uc_image = cast_to_unsigned_char(image)
    otsu_filter = itk.OtsuThresholdImageFilter[uc_image, uc_image].New()
    otsu_filter.SetInput(uc_image)
    otsu_filter.Update()
    return otsu_filter.GetOutput()


def multi_otsu_thresholding(
    image: itk.Image, number_of_thresholds: int, number_of_histogram_bins: int
) -> itk.Image:
    # Initialize the OtsuThresholdImageFilter
    uc_image = cast_to_unsigned_char(image)
    otsu_filter = itk.OtsuMultipleThresholdsImageFilter[uc_image, uc_image].New()
    otsu_filter.SetInput(uc_image)
    otsu_filter.SetNumberOfHistogramBins(number_of_histogram_bins)
    otsu_filter.SetNumberOfThresholds(number_of_thresholds)
    otsu_filter.Update()
    return otsu_filter.GetOutput()


def calculate_distance_between_points(
    point1: tuple[int, int, int], point2: tuple[int, int, int]
) -> float:
    x_distance = point1[0] - point2[0]
    y_distance = point1[1] - point2[1]
    z_distance = point1[2] - point2[2]
    distance = (x_distance**2 + y_distance**2 + z_distance**2) ** 0.5
    return distance


def plot_dict_images(
    threshold_dict: dict, center_dict: dict, true_center: tuple[int, int, int]
):
    SAVE_IMAGES = False

    for key, value in threshold_dict.items():
        center = center_dict[key]
        plt.imshow(value[5, :, :], cmap="gray")
        plt.scatter(center[0], center[1], c="r", marker="*")
        plt.scatter(true_center[0], true_center[1], c="b", marker="x")
        plt.title(f"Threshold: {key}")
        if SAVE_IMAGES:
            plt.savefig(f"Threshold_{key}.png")
        plt.show()

    # for key, value in threshold_dict.items():
    #
    #     plt.imshow(value[5,:,:],cmap="gray")
    #     plt.title(f"Threshold: {key}")
    #     plt.show()


def find_closest_center(true_center: tuple[int, int, int], center_dict: dict):
    closest_center = None
    closest_distance = float("inf")
    for key, value in center_dict.items():
        distance = calculate_distance_between_points(true_center, value)
        print(
            f"For {key}:"
            f"True Center: {true_center}, Center: {value}, Distance: {distance}"
            f"Closest Center: {closest_center}, Closest Distance: {closest_distance} for {key}"
            f"************************************"
        )
        if distance < closest_distance:
            closest_distance = distance
            closest_center = key
    return closest_center


def compare_average_center_of_mass():
    base_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/SortedProstateData"
    )
    output_data_path = Path("heck_Centers")
    output_data_path.mkdir(exist_ok=True, parents=True)

    thresholding_list = [1, 3, 5]
    histogram_list = [10, 30, 75]
    image_types = ["raw_image", "simple_otsu"] + [
        f"{t}_{h}"
        for t, h in itertools.product(thresholding_list, histogram_list)
        if t < h
    ]

    # Initialize dictionaries to store total distances and counts for each image type
    total_distances = {image_type: 0.0 for image_type in image_types}
    counts = {image_type: 0 for image_type in image_types}

    image_paths, mask_paths = init_data_lists(base_data_path)
    # selected_indices = random.sample(range(len(image_paths)), 10)
    # selected_image_paths = [image_paths[i] for i in selected_indices]
    # selected_mask_paths = [mask_paths[i] for i in selected_indices]
    selected_image_paths = image_paths
    selected_mask_paths = mask_paths
    for image_path, mask_path in zip(selected_image_paths, selected_mask_paths):
        print(f"Processing {image_path} and {mask_path}")
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)

        # Raw image
        raw_image_center = find_center_of_gravity(image)
        mask_center = find_center_of_gravity(mask)
        raw_distance = calculate_distance_between_points(raw_image_center, mask_center)
        total_distances["raw_image"] += raw_distance
        counts["raw_image"] += 1

        # Simple Otsu thresholding
        simple_otsu_image = simple_otsu_thresholding(image)
        simple_otsu_center = find_center_of_gravity(simple_otsu_image)
        simple_otsu_distance = calculate_distance_between_points(
            simple_otsu_center, mask_center
        )
        total_distances["simple_otsu"] += simple_otsu_distance
        counts["simple_otsu"] += 1

        # Multi Otsu thresholding
        for threshold, histogram in itertools.product(
            thresholding_list, histogram_list
        ):
            if threshold >= histogram:
                continue
            image_type = f"{threshold}_{histogram}"
            threshed_image = multi_otsu_thresholding(image, threshold, histogram)
            threshed_center = find_center_of_gravity(threshed_image)
            distance = calculate_distance_between_points(threshed_center, mask_center)
            total_distances[image_type] += distance
            counts[image_type] += 1

    # Calculate and print running average distances
    average_distances = {
        image_type: total_distances[image_type] / counts[image_type]
        for image_type in image_types
    }
    for image_type, avg_distance in average_distances.items():
        print(f"{image_type}: Average Distance = {avg_distance}")


if __name__ == "__main__":
    compare_average_center_of_mass()
