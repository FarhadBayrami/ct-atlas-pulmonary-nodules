import SimpleITK as sitk
import numpy as np
import os
import glob
import argparse
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def create_visualizations(input_folder: str, template_path: str, output_folder: str):
    logging.info("Starting visualization script...")
    
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        logging.info(f"Loading template image from: {template_path}")
        template_image = sitk.ReadImage(template_path, sitk.sitkFloat32)
        template_array = sitk.GetArrayFromImage(template_image)

        search_pattern = os.path.join(input_folder, '*WarpedToTemplate.nii.gz')
        image_paths = sorted(glob.glob(search_pattern))

        if not image_paths:
            logging.error(f"No warped images found in '{input_folder}'. Please check the path.")
            return

        logging.info(f"Found {len(image_paths)} images to process.")
        
        z_slice_idx = template_array.shape[0] // 2
        y_slice_idx = template_array.shape[1] // 2
        x_slice_idx = template_array.shape[2] // 2

        for img_path in image_paths:
            base_name = os.path.basename(img_path)
            logging.info(f"Processing '{base_name}'...")
            
            try:
                warped_image = sitk.ReadImage(img_path, sitk.sitkFloat32)
                
                resampler = sitk.ResampleImageFilter()
                resampler.SetReferenceImage(template_image)
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                resampled_image = resampler.Execute(warped_image)
                
                resampled_array = sitk.GetArrayFromImage(resampled_image)

                fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
                fig.suptitle(f'Multi-View Visualization: {base_name.replace(".nii.gz", "")}', color='white', fontsize=16)

                views = {
                    'Axial': (template_array[z_slice_idx, :, :], resampled_array[z_slice_idx, :, :]),
                    'Coronal': (template_array[:, y_slice_idx, :], resampled_array[:, y_slice_idx, :]),
                    'Sagittal': (np.rot90(template_array[:, :, x_slice_idx]), np.rot90(resampled_array[:, :, x_slice_idx]))
                }

                for i, (title, (template_slice, overlay_slice)) in enumerate(views.items()):
                    axes[i].imshow(template_slice, cmap='gray')
                    axes[i].imshow(np.ma.masked_where(overlay_slice == 0, overlay_slice), cmap='hot', alpha=0.6)
                    axes[i].set_title(title, color='white')
                    axes[i].axis('off')
                
                output_png_path = os.path.join(output_folder, base_name.replace('.nii.gz', '_visualization.png'))
                plt.savefig(output_png_path, bbox_inches='tight', dpi=150, facecolor='black')
                plt.close(fig)
                logging.info(f"✅ Successfully saved visualization to '{output_png_path}'")

            except Exception as e:
                logging.error(f"❌ Failed to process image '{base_name}': {e}")
                plt.close('all') # Ensure any open figures are closed on error
                continue
                
        logging.info("✅ All images processed. Script finished successfully.")

    except Exception as e:
        logging.error(f"❌ A critical error occurred: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create multi-view visualizations from resampled NIFTI images.")
    
    parser.add_argument(
        '--input_folder', 
        type=str,
        default='.',
        help="Path to the folder containing warped segmentation images."
    )
    
    parser.add_argument(
        '--template_path', 
        type=str, 
        required=True,
        help="Path to the template image file (e.g., BHI_template.nii.gz)."
    )
    
    parser.add_argument(
        '--output_folder', 
        type=str, 
        default='phase3_output',
        help="Path to the folder where output PNGs will be saved."
    )

    args = parser.parse_args()
    
    create_visualizations(args.input_folder, args.template_path, args.output_folder)