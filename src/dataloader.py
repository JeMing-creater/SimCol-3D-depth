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

def get_files(root, folders):
    maps = []
    imgs = []
    for vid in folders:
        files = sorted(glob.glob(vid + "*.png"))
        maps += [f for f in files if f.split("\\")[-1].startswith("Depth")]
        imgs += [f for f in files if f.split("\\")[-1].startswith("Frame")]
    imgs, maps = remove_bed_image(root, imgs, maps)
    return maps, imgs

def get_testloader(root, img_size=352, batch_size=1, num_workers=8):
    _, _, test_folders = get_train_test(root)
    test_maps, test_imgs = get_files(root, test_folders)
    
    transform_input = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    transform_target = transforms.ToTensor()

    test_dataset = Dataset(
        input_paths=test_imgs,
        target_paths=test_maps,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return test_dataloader

def get_dataloaders(root, img_size=352, batch_size=1, num_workers=8):

    train_folders, val_folders, _ = get_train_test(root)

    train_maps, train_imgs = get_files(root, train_folders)
    val_maps, val_imgs = get_files(root, val_folders)

    transform_input = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
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

def get_train_test(root):
    data_split_index = root + '/misc'

    train_split = data_split_index + '/train_file.txt'
    test_split = data_split_index + '/test_file.txt'
    val_split = data_split_index + '/test_file.txt'

    train_folders = []
    test_folders = []

    def add_folders(root, split):
        folders = []
        f = open(split,"r")             # 返回一个文件对象
        line = f.readline()             # 调用文件的 readline()方法
        while line:
            line = f.readline().strip('\n')
            folders.append(root + '/' + line)
        return folders
    
    train_folders = add_folders(root, train_split)
    test_folders = add_folders(root, test_split)
    val_folders = add_folders(root, val_split)


    return train_folders, test_folders, val_folders

def remove_bed_image(root, rgb, depth):
    # Remove bad frames
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png" in rgb:
        rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0059.png")
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png" in rgb:
        rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0060.png")
    if root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png" in rgb:
        rgb.remove(root + "/SyntheticColon_I/Frames_S14/FrameBuffer_0061.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0059.png" in depth:
        depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0059.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0060.png" in depth:
        depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0060.png")
    if root + "/SyntheticColon_I/Frames_S14/Depth_0061.png" in depth:
        depth.remove(root + "/SyntheticColon_I/Frames_S14/Depth_0061.png")
    return rgb, depth

if __name__ == "__main__":
    root = './datasets'

    img_size = 352
    batch_size=1
    train_loader, val_loader = get_dataloaders(root, img_size, batch_size, num_workers=0)

    test_loader = get_testloader(root, img_size, batch_size, num_workers=0)

    for (data, target) in train_loader:
        # print(data[0])
        print(data.shape)
        print(data.max())
        print(target.shape)
        print(target.max())
    
    for (data, target) in val_loader:
        # print(data[0])
        print(data.shape)
        print(data.max())
        print(target.shape)
        print(target.max())

    for (data, target) in test_loader:
        # print(data[0])
        print(data.shape)
        print(data.max())
        print(target.shape)
        print(target.max())