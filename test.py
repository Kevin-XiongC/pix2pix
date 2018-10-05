import cv2
import os
def format_image(img_path, size, nb_channels):
    """
    Load img with opencv and reshape
    """
    img = cv2.imread(img_path)

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    if nb_channels == 1:
        img = np.expand_dims(img, -1)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = np.expand_dims(img, 0).transpose(0, 3, 1, 2)

    return img
def build_HDF5(jpeg_dir, nb_channels, size=224):
    """
    Gather the data in a single HDF5 file.
    """

    # Put train data in HDF5
    file_name = os.path.basename(jpeg_dir.rstrip("/"))
    hdf5_file = os.path.join(data_dir, "%s_data.h5" % file_name)    
    with h5py.File(hdf5_file, "w") as hfw:
        for dset_type in ["train", "test", "val"]:

            list_img = [img for img in Path(jpeg_dir).glob('%s/img_*.jpg' % dset_type)]
            list_img = [str(img) for img in list_img]
           # list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
            list_img = list(map(str, list_img))
            list_img = np.array(list_img)

            list_img1 = [img for img in Path(jpeg_dir).glob('%s/dep_*.jpg' % dset_type)]
            list_img1 = [str(img) for img in list_img1]
           # list_img.extend(list(Path(jpeg_dir).glob('%s/*.png' % dset_type)))
            list_img1 = list(map(str, list_img1))
            list_img1 = np.array(list_img1)


            data_full = hfw.create_dataset("%s_data_full" % dset_type,
                                           (0, nb_channels, size, size),
                                           maxshape=(None, 3, size, size),
                                           dtype=np.uint8)

            data_sketch = hfw.create_dataset("%s_data_sketch" % dset_type,
                                             (0, nb_channels, size, size),
                                             maxshape=(None, 3, size, size),
                                             dtype=np.uint8)

            num_files = len(list_img)
            chunk_size = 100
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):

                list_img_path = list_img[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, nb_channels, pm_parallel=False)

                arr_img_full = np.concatenate([o for o in output], axis=0)

                # Resize HDF5 dataset
                data_full.resize(data_full.shape[0] + arr_img_full.shape[0], axis=0)
                data_full[-arr_img_full.shape[0]:] = arr_img_full.astype(np.uint8)

            num_files = len(list_img1)
            chunk_size = 100
            num_chunks = num_files / chunk_size
            arr_chunks = np.array_split(np.arange(num_files), num_chunks)

            for chunk_idx in tqdm(arr_chunks):
                list_img_path = list_img1[chunk_idx].tolist()
                output = parmap.map(format_image, list_img_path, size, nb_channels, pm_parallel=False)

                arr_img_dep = np.concatenate([o for o in output], axis=0)

                # Resize HDF5 dataset
                data_dep.resize(data_dep.shape[0] + arr_img_dep.shape[0], axis=0)
                data_dep[-arr_img_dep.shape[0]:] = arr_img_dep.astype(np.uint8)
