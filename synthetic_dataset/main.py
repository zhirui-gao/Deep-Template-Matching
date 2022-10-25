import synthetic_dataset
from pathlib import Path
import torch
import numpy as np
import torch.utils.data as data
from tqdm import tqdm
import cv2
import tarfile
import shutil
from imageio import imread

import random
import numpy as np
import cv2

def TwoPointRandInterP(p1,p2):
    v = (p2[0]-p1[0], p2[1]-p1[1])
    rr = random.random()
    v = (v[0]*rr, v[1]*rr)
    v = (int(v[0]+0.5), int(v[1]+0.5))
    return (p1[0]+v[0], p1[1]+v[1])


def positive_or_negative():
  if random.random() < 0.5:
    return 1
  else:
    return -1

def get_newpoint_between(p1, p2):
  v = (p2[0] - p1[0], p2[1] - p1[1])
  rr = random.random()
  v = (v[0] * rr, v[1] * rr)

  return ( round(p1[0]+positive_or_negative() * v[0]), round(p1[1]+ positive_or_negative() * v[1]))


def RPTransform(w,h):

    c_c = 0.05*w
    d_d = 0.05*h

    A = (0,0)
    B = (0,h-1)
    C = (w-1,h-1)
    D = (w-1,0)

    a = (c_c,d_d)
    b = (c_c, h-1-d_d)
    c = (w-1-c_c, h-1-d_d)
    d = (w-1-c_c, d_d)

    At = get_newpoint_between(A, a)
    Bt = get_newpoint_between(B, b)
    Ct = get_newpoint_between(C, c)
    Dt = get_newpoint_between(D, d)

    Pts0 = np.array([(0, 0), (0, h), (w, h), (w, 0)]).astype('float32')
    Pts1 = np.array([At, Bt, Ct, Dt]).astype('float32')
    H = cv2.getPerspectiveTransform(Pts0, Pts1)
    return H


def warp_image_Homo(path,H=480,W=640):
  Homo = RPTransform(w=H,h=W)
  img = cv2.imread(path)

  img_homo = cv2.warpPerspective(img, Homo, (W, H))
  homo_path = path.replace('.png', '_homo.png')
  cv2.imwrite(homo_path, img_homo)

  path_trans = path.replace('images', 'trans')
  path_trans = path_trans.replace('.png', '_trans.npy')
  M = np.load(path_trans)
  Homo_M = np.matmul(Homo, M)
  Homo_M = Homo_M/Homo_M[2][2] # template -> img
  path_trans_homo = path_trans.replace('trans.npy', 'trans_homo.npy')
  np.save(path_trans_homo, Homo_M)
  return Homo_M





TMPDIR = '/home/gzr/Data/synthetic_dataset_transfer/' # you can define your tmp dir

class SyntheticDataset_gaussian(data.Dataset):

    default_config = {
        "primitives": "all",
        "truncate": {},
        "validation_size": -1,
        "test_size": -1,
        "on-the-fly": False,
        "cache_in_memory": False,
        "suffix": None,
        "add_augmentation_to_test_set": False,
        "num_parallel_calls": 10,
        "generation": {
            "split_sizes": {"training": 0, "validation": 0, "test": 50},
            "image_size": [480, 640],
            "random_seed": 0,
            "params": {
                "generate_background": {
                    "min_kernel_size": 150,
                    "max_kernel_size": 500,
                    "min_rad_ratio": 0.02,
                    "max_rad_ratio": 0.031,
                },
                "draw_stripes": {"transform_params": (0.1, 0.1)},
                "draw_multiple_polygons": {"kernel_boundaries": (50, 100)},
            },
        },
        "preprocessing": {"resize": [480, 640], "blur_size": 5,},
        "augmentation": {
            "photometric": {
                "enable": False,
                "primitives": "all",
                "params": {},
                "random_order": True,
            },
            "homographic": {"enable": False, "params": {}, "valid_border_margin": 0,},
        },
    }

    def __init__(self):
        self.config = self.default_config
        DATA_PATH = 'synthetic_dataset'
        basepath = Path(
            DATA_PATH,
            "synthetic_shapes"
            + ("_{}".format(self.config["suffix"]) if self.config["suffix"] is not None else ""),
        )
        basepath.mkdir(parents=True, exist_ok=True)

        primitives = ['draw_ellipses', 'draw_contours','draw_star','draw_polygon'] #,'draw_ellipses', 'draw_contours',
        for primitive in primitives:
            tar_path = Path(basepath, "{}.tar.gz".format(primitive))
            # if not tar_path.exists():
            self.dump_primitive_data(primitive, tar_path, self.config)

    def get_external_contours_points_sample(self, image,num_point):
        """
        :param image: (H,W)
        :return: (N,2)
        """
        assert (len(image.shape) == 2)
        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        xcnts = np.vstack([x.reshape(-1, 2) for x in contours])

        gap = xcnts.shape[0] // num_point
        xcnts = xcnts[:num_point * gap:gap, :]
        return xcnts


    def dump_primitive_data(self, primitive, tar_path, config):
        # temp_dir = Path(os.environ['TMPDIR'], primitivconfig = {dict: 15} {'primitives': 'all', 'truncate': {'draw_ellipses': 0.3, 'draw_stripes': 0.2, 'gaussian_noise': 0.1}, 'validation_size': -1, 'test_size': -1, 'on-the-fly': False, 'cache_in_memory': True, 'suffix': 'v6', 'add_augmentation_to_test_set': False, 'num_parallel… Viewe)
        temp_dir = Path(TMPDIR, primitive)

        # tf.logging.info("Generating tarfile for primitive {}.".format(primitive))
        synthetic_dataset.set_random_state(
            np.random.RandomState(config["generation"]["random_seed"])
        )
        for split, size in self.config["generation"]["split_sizes"].items():
            im_dir, pts_dir,trans_dir = [Path(temp_dir, i, split) for i in ["images", "points", "trans"]]
            im_dir.mkdir(parents=True, exist_ok=True)
            pts_dir.mkdir(parents=True, exist_ok=True)
            trans_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(size), desc=split, leave=False):
                # image = synthetic_dataset.generate_background(
                #     config["generation"]["image_size"],
                #     **config["generation"]["params"]["generate_background"],
                # )
                image = synthetic_dataset.generate_background_steel('/home/gzr/Data/generative_steel/train',config["generation"]["image_size"],10000)
                image_back = np.copy(image)

                points = None
                while points is None:
                    points, template_points, template_img,tran_matrix = np.array(
                        getattr(synthetic_dataset, primitive)(
                            image, **config["generation"]["params"].get(primitive, {})
                        )
                    )

                points = np.flip(points, 1)  # reverse convention with opencv (x,y)->(y,x)
                template_points = np.flip(template_points, 1)  # reverse convention with opencv

                b = config["preprocessing"]["blur_size"]
                image = cv2.GaussianBlur(image, (b, b), 0)
                points = (
                        points
                        * np.array(config["preprocessing"]["resize"], np.float)
                        / np.array(config["generation"]["image_size"], np.float)
                )
                template_points = (
                        template_points
                        * np.array(config["preprocessing"]["resize"], np.float)
                        / np.array(config["generation"]["image_size"], np.float)
                )

                image = cv2.resize(
                    image,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )
                template_img = cv2.resize(
                    template_img,
                    tuple(config["preprocessing"]["resize"][::-1]),
                    interpolation=cv2.INTER_LINEAR,
                )

                cv2.imwrite(str(Path(im_dir, "{}.png".format(i))), image*0.6+image_back*0.4)
                cv2.imwrite(str(Path(im_dir, "{}_template.png".format(i))), template_img)

                np.save(Path(trans_dir, "{}_trans.npy".format(i)), tran_matrix)

                tran_matrix_homo = warp_image_Homo(str(Path(im_dir, "{}.png".format(i))))

                # find contures in template img
                template_points = self.get_external_contours_points_sample(template_img, 20)


                np.save(Path(pts_dir, "{}_template.npy".format(i)), template_points)

                x = template_points[:, 0] * tran_matrix_homo[0, 0] + template_points[:, 1] * tran_matrix_homo[0, 1] + tran_matrix_homo[
                    0, 2]
                y = template_points[:, 0] * tran_matrix_homo[1, 0] + template_points[:, 1] * tran_matrix_homo[1, 1] + tran_matrix_homo[
                    1, 2]
                z = template_points[:, 0] * tran_matrix_homo[2, 0] + template_points[:, 1] * tran_matrix_homo[2, 1] + tran_matrix_homo[
                    2, 2]
                points = np.concatenate(((x / z).reshape(-1,1), (y / z).reshape(-1,1)), axis=1)
                np.save(Path(pts_dir, "{}.npy".format(i)), points)



        # Pack into a tar file
        # tar = tarfile.open(tar_path, mode="w:gz")
        # tar.add(temp_dir, arcname=primitive)
        # tar.close()
        # shutil.rmtree(temp_dir)
        # tf.logging.info("Tarfile dumped to {}.".format(tar_path))




SyntheticDataset_gaussian()