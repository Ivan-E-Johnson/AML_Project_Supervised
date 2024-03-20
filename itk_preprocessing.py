import itk
from pathlib import Path

from matplotlib import pyplot as plt

from data_connector import get_exploratory_image_info
import pandas as pd


def show_images(image, mask, num_images=1):
    fig, ax = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    for i in range(num_images):
        image_arr = itk.array_from_image(image)
        mask_arr = itk.array_from_image(mask)
        ax[i, 0].imshow(image_arr[5], cmap="gray")
        ax[i, 0].set_title("Image")
        ax[i, 1].imshow(mask_arr[5])
        ax[i, 1].set_title("Mask")
    plt.tight_layout()
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


def calculate_bounding_box_extent(
    bounding_box: tuple[tuple[int, int, int], tuple[int, int, int]]
) -> tuple[int, int, int]:
    x_extent = bounding_box[1][0] - bounding_box[0][0]
    y_extent = bounding_box[1][1] - bounding_box[0][1]
    z_extent = bounding_box[1][2] - bounding_box[0][2]
    print(f"X Extent: {x_extent}, Y Extent: {y_extent}, Z Extent: {z_extent}")
    return x_extent, y_extent, z_extent


def convert_multiclass_mask_to_binary(mask: itk.Image) -> itk.Image:
    # Initialize the BinaryThresholdImageFilter
    binary_filter = itk.BinaryThresholdImageFilter.New(
        Input=mask, LowerThreshold=0, UpperThreshold=0, InsideValue=1, OutsideValue=0
    )
    binary_filter.Update()

    return binary_filter.GetOutput()


if __name__ == "__main__":
    base_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/SortedProstateData"
    )
    output_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/PreProcessedProstateData"
    )
    output_data_path.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame()
    image_paths, mask_paths = init_data_lists(base_data_path)
    for image_path, mask_path in zip(image_paths, mask_paths):
        print(image_path, mask_path)
        image = itk.imread(str(image_path), itk.F)
        mask = itk.imread(str(mask_path), itk.UC)
        image_info = get_exploratory_image_info(image)
        mask_info = get_exploratory_image_info(mask)
        image_info["image_name"] = image_path.name
        mask_info["mask_name"] = mask_path.name
        if df.empty:
            df = pd.DataFrame.from_dict(image_info)
        else:
            df = pd.concat([df, pd.DataFrame.from_dict(image_info)], ignore_index=True)
        print(f"Image Info: {image_info}")
        print(f"Mask Info: {mask_info}")
        show_images(image, mask)
        # Convert the multiclass mask to a binary mask
        binary_mask = convert_multiclass_mask_to_binary(mask)

        show_images(image, binary_mask)
        # Get the bounding box for the prostate
        prostate_bounding_box = get_mask_bounding_box(mask, 1)

        # Calculate the extent of the bounding box
        prostate_extent = calculate_bounding_box_extent(prostate_bounding_box)

        break
    print(df.head())
    df.to_csv("image_info.csv", index=False)

    print(image_paths)
