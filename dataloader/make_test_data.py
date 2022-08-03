import argparse
import os
from PIL import Image
import shutil


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
        os.makedirs(args.dst_dir)

    dirs = os.listdir(args.src_dir)

    print("-- processing images...")

    file_count_full = sum([len(files) for r, d, files in os.walk(args.src_dir)])
    file_count_current = 0

    for subdir, dirs, files in os.walk(args.src_dir):

        subdir_without_root = subdir.replace(args.src_dir, "")
        current_new_subdir = f"{args.dst_dir}\\{subdir_without_root}"

        if not os.path.isdir(current_new_subdir):
            os.makedirs(current_new_subdir)

        for file in files:
            # if the file is not an image, just copy it

            filename, file_extension = os.path.splitext(os.path.join(subdir, file))
            if file_extension not in [".jpg", ".jpeg", ".png", ".gif"]:
                shutil.copyfile(f"{subdir}\\{file}", f"{current_new_subdir}\\{file}")
                continue

            file_count_current += 1
            im = Image.open(os.path.join(subdir, file))

            # make non-square images square
            if im.width != im.height:
                new_size = min(im.width, im.height)

                left = int((im.width - new_size) / 2)
                top = int((im.height - new_size) / 2)
                right = int((im.width + new_size) / 2)
                bottom = int((im.height + new_size) / 2)

                im = im.crop((left, top, right, bottom))

            im_resized = im.resize((args.img_size, args.img_size))
            im_resized.save(f"{current_new_subdir}\\{file}")

            if file_count_current % 500 == 0:
                print(f"-- processed {file_count_current}/{file_count_full} images")

    print("-- finished processing images!")
