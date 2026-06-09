import SimpleITK as sitk
import os
import glob
import argparse
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def sanitize_filenames(input_folder: str):
    logging.info("Sanitizing filenames...")
    renamed_count = 0
    
    # Pattern 1: *.gz -> *.nii.gz
    pattern1 = os.path.join(input_folder, '*WarpedToTemplate.gz')
    for old_path in glob.glob(pattern1):
        if "_nii.gz" not in old_path: # Avoid double renaming
            new_path = old_path.replace('.gz', '.nii.gz')
            try:
                os.rename(old_path, new_path)
                logging.info(f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
                renamed_count += 1
            except OSError as e:
                logging.warning(f"Could not rename {old_path}: {e}")

    # Pattern 2: *_nii.gz -> *.nii.gz
    pattern2 = os.path.join(input_folder, '*_nii.gz')
    for old_path in glob.glob(pattern2):
        new_path = old_path.replace('_nii.gz', '.nii.gz')
        try:
            os.rename(old_path, new_path)
            logging.info(f"Renamed: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
            renamed_count += 1
        except OSError as e:
            logging.warning(f"Could not rename {old_path}: {e}")
            
    if renamed_count > 0:
        logging.info("Filename sanitization complete.")
    else:
        logging.info("All filenames appear to be correct. No changes made.")


def generate_frequency_map(input_folder: str, output_file: str):
    try:
        sanitize_filenames(input_folder)

        logging.info("Starting frequency map generation process...")
        
        search_pattern = os.path.join(input_folder, '*WarpedToTemplate.nii.gz')
        image_paths = sorted(glob.glob(search_pattern))

        if not image_paths:
            logging.error(f"No valid NIFTI images found with pattern '{search_pattern}'. Please check file names and path.")
            return

        logging.info(f"Found {len(image_paths)} images to process.")

        logging.info(f"Using '{os.path.basename(image_paths[0])}' as reference image.")
        reference_image = sitk.ReadImage(image_paths[0], sitk.sitkFloat64)
        
        accumulator = sitk.Image(reference_image.GetSize(), sitk.sitkFloat64)
        accumulator.CopyInformation(reference_image)

        for path in image_paths:
            logging.info(f"Processing: {os.path.basename(path)}")
            
            current_image = sitk.ReadImage(path, sitk.sitkFloat64)
            
            resampled_image = sitk.Resample(current_image, reference_image, sitk.Transform(), 
                                            sitk.sitkNearestNeighbor, 0.0, current_image.GetPixelID())
            
            accumulator = sitk.Add(accumulator, resampled_image)

        output_image = sitk.Cast(accumulator, sitk.sitkInt16)
        
        logging.info(f"Saving final frequency map to: '{output_file}'")
        sitk.WriteImage(output_image, output_file)
        
        logging.info("✅ Process completed successfully.")

    except Exception as e:
        logging.error(f"❌ An error occurred during the script execution: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates an absolute frequency map from NIFTI segmentation images.")
    
    parser.add_argument(
        '--input_folder', 
        type=str, 
        default='.', 
        help="Path to the folder containing input images. Defaults to the current folder."
    )
    
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='frequency_map.nii.gz',
        help="Name and path for the output frequency map file. Defaults to 'frequency_map.nii.gz'."
    )

    args = parser.parse_args()
    
    generate_frequency_map(args.input_folder, args.output_file)