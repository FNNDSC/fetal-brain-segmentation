import tqdm
import numpy as np
import nibabel as nib

from mrcnn import utils
from data_loader import normalize_0_255

class FBSDataset(utils.Dataset):

    def load_data(self, data_files):
        image_files, mask_files = data_files

        self.add_class('MRI', 1, 'brain')
        img_id = 0

        for i in tqdm.tqdm(range(len(image_files)), desc='loading'):

            img_path = image_files[i]
            mask_path = mask_files[i]

            img_slices = nib.load(img_path)
            mask_slices = nib.load(mask_path)

            img_slices = img_slices.get_fdata()
            mask_slices = mask_slices.get_data()

            for j in range(img_slices.shape[-1]):
                img = np.array(img_slices[:,:,j])
                mask = np.array(mask_slices[:,:,j])

                new_img = normalize_0_255(img)

                mask = mask[..., np.newaxis]
                new_img = new_img[..., np.newaxis]
                mask = np.array(mask, dtype=np.uint16) * 255
                new_img = np.array(new_img, dtype=np.uint16)

                self.add_image('MRI',
                        image=new_img,
                        shape=new_img.shape,
                        mask=mask,
                        mask_shape=mask.shape,
                        image_id=i,
                        path=img_path,
                        mask_path=mask_path,
                        width=img.shape[0],
                        height=img.shape[1])

                img_id += 1
        return img_id

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def load_mask(self, image_id):
        mask = self.image_info[image_id]['mask']
        return mask, np.ones([1]).astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        return self.image_info[image_id]
