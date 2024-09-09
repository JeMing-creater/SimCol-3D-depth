
SimCol3D

This dataset and accompanying files are licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). https://creativecommons.org/licenses/by-nc-sa/4.0/




1. Dataset
_________________

This dataset comprises three synthetic colonoscopy scenes (I-III) and a folder with miscellaneous files. The data is arranged as follows:

.
├── misc
├── SyntheticColon_I
│   ├── Frames_S1
│   ├── Frames_S10
│   ├── Frames_S11
│   ├── Frames_S12
│   ├── Frames_S13
│   ├── Frames_S14
│   ├── Frames_S15
│   ├── Frames_S2
│   ├── Frames_S3
│   ├── Frames_S4
│   ├── Frames_S5
│   ├── Frames_S6
│   ├── Frames_S7
│   ├── Frames_S8
│   └── Frames_S9
├── SyntheticColon_II
│   ├── Frames_B1
│   ├── Frames_B10
│   ├── Frames_B11
│   ├── Frames_B12
│   ├── Frames_B13
│   ├── Frames_B14
│   ├── Frames_B15
│   ├── Frames_B2
│   ├── Frames_B3
│   ├── Frames_B4
│   ├── Frames_B5
│   ├── Frames_B6
│   ├── Frames_B7
│   ├── Frames_B8
│   └── Frames_B9
└── SyntheticColon_III
    ├── Frames_O1
    ├── Frames_O2
    └── Frames_O3

The cam.txt file in each of the three scene folders includes camera intrinsics. We provide the official training and testing splits for the MICCAI challenge in the respective txt files in the misc folder.



2. Poses and depths
_________________

Each subfolder "Frames_xx" consists of 601 or 1201 RGB images and depth maps. The greyscale depth images in the range [0, 1] correspond to [0, 20] cm in world space. Please see the challenge GitHub page for details on how to load and visualize the data: https://github.com/anitarau/simcol/blob/main/data_helpers/visualize_3D_data.py

The respective camera poses are provided in .txt files inside the scene folders. "SavedPosition" contains the camera translations in the following format: tx, ty, tz. "SavedRotationQuaternion" includes the camera rotations as quaternions in the following format: qx, qy, qz, qw. Here, t denotes the translation, and q the rotation as quaternions. Each text file corresponds to one folder and can be matched based on the trajectory index (1-15). 

To obtain the relative pose between two images, we recommend the package scipy.spatial.transform.Rotation to translate between quaternions and rotation matrices. Note that the camera poses are provided in a left-handed coordinate system while the Rotation package assumes right-handedness. Please find more details in the misc/read_poses.py file. 



3. References
_________________

The synthetic data generation process and SyntheticColon_I was developed for the publication:

@article{rau2022bimodal,
  title={Bimodal camera pose prediction for endoscopy},
  author={Rau, Anita and Bhattarai, Binod and Agapito, Lourdes and Stoyanov, Danail},
  journal={arXiv preprint arXiv:2204.04968},
  year={2022}
}


SyntheticColon_II and SyntheticColon_III were added for the SimCol3D MICCAI challenge and the publication:  


@article{rau2023simcol3d,
  title={SimCol3D--3D Reconstruction during Colonoscopy Challenge},
  author={Rau, Anita and Bano, Sophia and Jin, Yueming and Azagra, Pablo and Morlana, Javier and Sanderson, Edward and Matuszewski, Bogdan J and Lee, Jae Young and Lee, Dong-Jae and Posner, Erez and others},
  journal={arXiv preprint arXiv:2307.11261},
  year={2023}
}


Please cite our work accordingly if you found it helpful. 



4. Contact
_________________

The information in this README was provided by Anita Rau on 24. August 2023. If you have any questions, please email a.rau.16@ucl.ac.uk.

