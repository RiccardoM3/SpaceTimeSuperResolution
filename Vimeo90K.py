import os
import cv2

class Vimeo90K:
    def __init__(self):
        pass

    def save_downscaled_image(self, image_path, downscaled_image_path):
        img = cv2.imread(image_path)
        scaled_img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(downscaled_image_path, scaled_img)

    def prepare_data(self):
        sequences_folder = os.path.join('vimeo_septuplet', 'sequences')
        hr_sequences_folder = os.path.join('vimeo_septuplet', 'hr_sequences')
        lr_sequences_folder = os.path.join('vimeo_septuplet', 'lr_sequences')

        # First rename the sequences folder to hr_sequences
        os.rename(sequences_folder, hr_sequences_folder)
        
        # Now create an lr_sequences folder
        os.makedirs(lr_sequences_folder)

        print("Generating LR Images...")
        for sequence_folder in os.listdir(hr_sequences_folder):
            hr_sequence_folder_path = os.path.join(hr_sequences_folder, sequence_folder)
            lr_sequence_folder_path = os.path.join(lr_sequences_folder, sequence_folder)

            print(f"Generating images for {lr_sequence_folder_path}...")
            for sub_sequence_folder in os.listdir(hr_sequence_folder_path):
                hr_sub_sequence_folder_path = os.path.join(hr_sequence_folder_path, sub_sequence_folder)
                lr_sub_sequence_folder_path = os.path.join(lr_sequence_folder_path, sub_sequence_folder)
                os.makedirs(lr_sub_sequence_folder_path, exist_ok=True)
                
                for image_file in os.listdir(hr_sub_sequence_folder_path):
                    hr_image_path = os.path.join(hr_sub_sequence_folder_path, image_file)
                    lr_image_path = os.path.join(lr_sub_sequence_folder_path, image_file)
                    self.save_downscaled_image(hr_image_path, lr_image_path)