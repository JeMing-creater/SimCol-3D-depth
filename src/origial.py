import numpy as np
import random
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data
import torchvision.transforms.functional as TF

class Dataset(data.Dataset):
    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input=None,
        transform_target=None,
        hflip=False,
        vflip=False,
        affine=False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]
        target_ID = self.target_paths[index]

        x = np.array(Image.open(input_ID))[:, :, :3]
        y = np.array(Image.open(target_ID)) / 255 / 256

        x = self.transform_input(x)
        y = self.transform_target(y)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-352 / 8, 352 / 8)
            v_trans = random.uniform(-352 / 8, 352 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(x, angle, (h_trans, v_trans), scale, shear, fill=-1.0)
            y = TF.affine(y, angle, (h_trans, v_trans), scale, shear, fill=0.0)
        return x.float(), y.float()


# class Dataset_test(data.Dataset):
#     def __init__(self, input_paths: list, transform_input=None):
#         self.input_paths = input_paths
#         self.transform_input = transform_input

#     def __len__(self):
#         return len(self.input_paths)

#     def __getitem__(self, index: int):
#         input_ID = self.input_paths[index]

#         x = np.array(Image.open(input_ID))[:, :, :3]
#         x = self.transform_input(x)

#         return x.float()

def get_dataloaders(train_maps, train_imgs, val_maps, val_imgs, batch_size, num_workers=8):
    
    transform_input = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((352, 352), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.ToTensor()

    train_dataset = Dataset(
        input_paths=train_imgs,
        target_paths=train_maps,
        transform_input=transform_input,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=False,
    )

    val_dataset = Dataset(
        input_paths=val_imgs,
        target_paths=val_maps,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_dataloader, val_dataloader

def get_val_test(root):
    val_vids = [
        root + "/SyntheticColon_I/Frames_S4/",
        root + "/SyntheticColon_I/Frames_S9/",
        root + "/SyntheticColon_I/Frames_S14/",
        root + "/SyntheticColon_II/Frames_B4/",
        root + "/SyntheticColon_II/Frames_B9/",
        root + "/SyntheticColon_II/Frames_B14/",
    ]
    test_vids = [
        root + "/SyntheticColon_I/Frames_S5/",
        root + "/SyntheticColon_I/Frames_S10/",
        root + "/SyntheticColon_I/Frames_S15/",
        root + "/SyntheticColon_II/Frames_B5/",
        root + "/SyntheticColon_II/Frames_B10/",
        root + "/SyntheticColon_II/Frames_B15/",
        root + "/SyntheticColon_III/Frames_O1/",
        root + "/SyntheticColon_III/Frames_O2/",
        root + "/SyntheticColon_III/Frames_O3/",
    ]
    return val_vids, test_vids

def remove_bed_image(root, val_rgb, val_depth):
    # Remove bad frames
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png" in val_rgb:
        val_rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png")
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png" in val_rgb:
        val_rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png")
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png" in val_rgb:
        val_rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0059.png" in val_depth:
        val_depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0059.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0060.png" in val_depth:
        val_depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0060.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0061.png" in val_depth:
        val_depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0061.png")
    return val_rgb, val_depth

if __name__ == "__main__":
    root = './datasets'
    base_folders = sorted(glob.glob(root + "/*/"))
    sub_folders = []
    for bf in base_folders:
        sub_folders += sorted(glob.glob(bf + "*/"))
    
    val_vids, test_vids = get_val_test(root)

    train_vids = sub_folders
    for vid in test_vids + val_vids:
        if vid in train_vids:
            train_vids.remove(vid)
    
    train_depth = []
    train_rgb = []
    val_depth = []
    val_rgb = []

    for vid in train_vids:
        files = sorted(glob.glob(vid + "*.png"))
        train_depth += [f for f in files if f.split("\\")[-1].startswith("Depth")]
        train_rgb += [f for f in files if f.split("\\")[-1].startswith("Frame")]
    for vid in val_vids:
        files = sorted(glob.glob(vid + "*.png"))
        val_depth += [f for f in files if f.split("\\")[-1].startswith("Depth")]
        val_rgb += [f for f in files if f.split("\\")[-1].startswith("Frame")]

    val_rgb, val_depth = remove_bed_image(root, val_rgb, val_depth)

    batch_size=1
    train_loader, val_loader = get_dataloaders(train_depth, train_rgb, val_depth, val_rgb, batch_size, num_workers=0)

    for (data, target) in train_loader:
        # print(data[0])
        print(data.shape)
        print(max(data))
        print(target.shape)
        print(max(target))