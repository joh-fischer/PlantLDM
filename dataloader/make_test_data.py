import argparse
import os
from PIL import Image


def get_argparser():
    """
    Initializes the argument parser
    :return: the argument parser object
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="directory in which the source images are stored"
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        required=True,
        help="directory in which the resulting images will be stored"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        required=True,
        help="Size of the resulting images. Images are supposed to be square."
    )

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()

    # check if directories exist
    if not os.path.isdir(args.src_dir):
        raise ValueError("src_dir does not exist!")
    if not os.path.isdir(args.dst_dir):
        raise ValueError("dst_dir does not exist!")

    dirs = os.listdir(args.src_dir)

    print("-- processing images...")

    file_count_full = sum([len(files) for r, d, files in os.walk(args.src_dir)])
    file_count_current = 0

    for subdir, dirs, files in os.walk(args.src_dir):

        subdir_without_root = subdir.replace(args.src_dir, "")
        # currently data is separated in test, train and val. We combine these and do the split later in the dataloader
        subdir_without_root = subdir_without_root.replace("images_test", "")
        subdir_without_root = subdir_without_root.replace("images_train", "")
        subdir_without_root = subdir_without_root.replace("images_val", "")
        current_new_subdir = f"{args.dst_dir}\\{subdir_without_root}"

        # for performance reasons we create subdirs even if they don't have files in them. Else we would have to
        # check this in the files loop
        if not os.path.exists(current_new_subdir):
            os.makedirs(current_new_subdir)

        for file in files:
            file_count_current += 1
            im = Image.open(os.path.join(subdir, file))
            im_resized = im.resize((args.img_size, args.img_size))  # TODO: maybe with Image.ANTIALIAS?
            im_resized.save(f"{current_new_subdir}\\{file}")

            if file_count_current % 500 == 0:
                print(f"-- processed {file_count_current}/{file_count_full} images")

    print("-- finished processing images!")