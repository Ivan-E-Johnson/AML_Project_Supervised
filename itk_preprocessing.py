import itk
from pathlib import Path

from matplotlib import pyplot as plt

from data_connector import get_exploratory_image_info, get_image_stats
import pandas as pd


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


if __name__ == "__main__":
    base_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/SortedProstateData"
    )
    output_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/PreProcessedProstateData"
    )
    output_data_path.mkdir(exist_ok=True, parents=True)

    image_paths, mask_paths = init_data_lists(base_data_path)
    write_exploratry_image_info(image_paths, mask_paths, output_data_path)