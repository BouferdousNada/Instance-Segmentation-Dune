"""
Mask R-CNN
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:


    # Train a new model starting from specific weights file
    python dunes.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python dunes.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Train a model with Edge Agreement Head
    python dunes.py train --dataset=dataset --subset=train --weights=<last or /path/to/weights.h5> --edge-loss-smoothing-predictions --edge-loss-smoothing-groundtruth --edge-loss-filters sobel --edge-loss-weight-factor 1.0 --edge-loss-weight-entropy

    # Generate submission file
    python dunes.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import datetime
import numpy as np
import skimage.io


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results/dunes/")

# The dataset doesn't have a standard train/val split, so I picked
# a variety of images to surve as a validation set.
VAL_IMAGE_IDS = [
    "G04_220817_100_AdditiveGaussianNoise",
    "G04_220817_100_Affine",
    "G04_220817_100_CLAHE",
    "G04_220817_100_Sequential_2",
    "G06_121118_100_Affine_0",
    "G06_121118_100_EnhanceBrightness",
    "G07_180619_100_Affine_1",
    "G07_180619_100_Fliplr",
    "G07_180619_100_Sequential_3",
    "G07_180619_100_TranslateX",
    "G08_230919_100_EnhanceSharpness",
    "G08_230919_100_Sequential_4",
    "G09_230919_100_CLAHE",
    "G09_230919_100_Sequential_2",
    "G09_230919_100_TranslateY",
    "G09_240913_100_Sequential",
    "G10_19_100",
    "G10_210914_100_Affine",
    "G10_230813_100_Affine_1",
    "G10_230813_100_Fliplr",
    "G11_270919_100_EnhanceSharpness",
    "G11_280919_100_Affine_0",
    "G11_290919_100_Flipud",
    "G11_300919_100_SaltAndPepper",
    "G12_140613_100_Sequential",
    "G12_140613_100_Sequential_4",
    "G12_250914_100_EnhanceBrightness",
    "G13_051014_100_Affine",
    "G13_051014_100_SaltAndPepper",
    "G13_051014_100_Sequential_3",
    "G13_051014_100_TranslateX",
    "G13_19_100_EnhanceSharpness",
    "G14_081012_100_Affine_0",
    "G14_081012_100_CLAHE",
    "G14_081012_100",
    "G14_081012_100_Flipud",
    "G14_19_100_AdditiveGaussianNoise",
    "G14_19_100_TranslateY",
    "G14_220917_100_AdditiveGaussianNoise",
    "G14_220917_100_EnhanceBrightness",
    "G14_220917_100_TranslateX",
    "G15_011014_100_Affine",
    "G15_011014_100_CLAHE",
    "G15_011014_100_Flipud",
    "G15_011014_100_Sequential_1",
    "G15_011014_100_TranslateY",
    "G15_031014_100_Fliplr",
    "G15_031014_100_SaltAndPepper",
    "G15_031014_100_Sequential_3",
    "G15_19_2_100_Affine",
    "G15_19_2_100_Sequential",
    "G15_19_3_100_Affine_0",
    "G15_19_3_100_EnhanceBrightness",
    "G15_19_3_100_TranslateY",
    "G15_19_3_100"
]

############################################################
#  Configurations
############################################################

class DunesConfig(Config):
    """Configuration for training on the dunes segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "dunes"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + dunes

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between dunes and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    BACKBONE = "resnet101"

    # Input image resizing (Random crops of size 512x512)
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000
    PRE_NMS_LIMIT = 6000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400

    def __init__(self, args=None):  # make args optional
        super().__init__()
        # -- EDGE AGREEMENT HEAD CONFIGURATION --
        # Use default values if args is None
        self.RUN_NAME = getattr(args, 'run_name', None)
        # EDGE_LOSS_SMOOTHING_GT: Apply smoothing to the **ground truth edges** before computing the edge loss.
        self.EDGE_LOSS_SMOOTHING_GT = getattr(args, 'edge_loss_smoothing_groundtruth', False)
        # Apply smoothing to the **predicted masks' edges** before computing the edge loss.
        self.EDGE_LOSS_SMOOTHING_PREDICTIONS = getattr(args, 'edge_loss_smoothing_predictions', False)
        # List of filters (e.g., Sobel, Laplacian) used to detect edges in the masks (Sobel)
        self.EDGE_LOSS_FILTERS = getattr(args, 'edge_loss_filters', [])
        # Type of norm (e.g., "l1", "l2") used to compute the difference between predicted and ground truth edges.
        self.EDGE_LOSS_NORM = getattr(args, 'edge_loss_norm', "l2")
        # Scalar multiplier to adjust the contribution of the edge loss relative to other losses (like mask loss or bounding box loss).
        self.EDGE_LOSS_WEIGHT_FACTOR = getattr(args, 'edge_loss_weight_factor', 1.0)
        # If True, dynamically adjusts the weight of the edge loss based on entropy (uncertainty) in the mask predictions.
        self.EDGE_LOSS_WEIGHT_ENTROPY = getattr(args, 'edge_loss_weight_entropy', False)
        # MASK_SHAPE: Defines the resolution of masks used for training the mask head (typically (28,28) but can be modified via args).
        self.MASK_SHAPE = (getattr(args, 'mask_size', 28), getattr(args, 'mask_size', 28))



class DunesInferenceConfig(DunesConfig):
    def __init__(self, args=None):
        super().__init__(args)  # Pass None or a default argument if args is not needed
    # Rest of the DunesInferenceConfig setup as defined in your original code
    EDGE_LOSS_FILTERS = ["sobel-x", "sobel-y"]
    EDGE_LOSS_NORM = "l2"
    EDGE_LOSS_WEIGHT_FACTOR = 1.0
    EDGE_LOSS_SMOOTHING_PREDICTIONS = True
    EDGE_LOSS_SMOOTHING_GT = True
    USE_MINI_MASK = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "pad64"
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################
class DunesDataset(utils.Dataset):

    def load_dunes(self, dataset_dir, subset):
        """Load a subset of the dune dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        self.add_class("dunes", 1, "dunes")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", "stage1_train", "stage1_test", "stage2_test"]
        subset_dir = "stage1_train" if subset in ["train", "val"] else subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
        # with open("log.txt", "w") as f:
        #     f.write("dataset" + dataset_dir)
        if subset == "val":
            image_ids = VAL_IMAGE_IDS
        else:
            # Get image ids from directory names
            image_ids = next(os.walk(dataset_dir))[1]
            if subset == "train":
                image_ids = list(set(image_ids) - set(VAL_IMAGE_IDS))

        # Add images
        for image_id in image_ids:
            self.add_image("dunes",
                image_id=image_id,
                path=os.path.join(dataset_dir, image_id,
                                  "mnt/{}_mnt.png".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        # with open("log.txt", "w") as f:
        #     f.write("mask_dir" + str(mask_dir))

        # Read mask files from .png image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".png"):
                m = skimage.io.imread(os.path.join(mask_dir, f))#.astype(np.bool)
                m=m[:,:,0]
                mask.append(m)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)
        


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "dunes":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = DunesDataset()
    dataset_train.load_dunes(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DunesDataset()
    dataset_val.load_dunes(dataset_dir, "val")
    dataset_val.prepare()

    def compute_total_loss(pred_mask, gt_mask):
        # Calculate existing Mask R-CNN loss
        total_loss = model.compute_loss(pred_mask, gt_mask)

        # Add edge agreement loss if enabled
        if config.EDGE_LOSS_FILTERS:
            edge_loss = model.edge_agreement_loss(
                pred_mask,
                gt_mask,
                filters=config.EDGE_LOSS_FILTERS,
                norm=config.EDGE_LOSS_NORM,
                weight_factor=config.EDGE_LOSS_WEIGHT_FACTOR
            )
            total_loss += edge_loss
        return total_loss

    print("Train layers from layer 3+")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs= 380,
                layers='3+')


############################################################
#  RLE Encoding
############################################################
def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################
def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = DunesDataset()
    dataset.load_dunes(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)




############################################################
#  Command Line
############################################################
if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--edge-loss-smoothing-predictions',
                        action='store_true',
                        help='Apply a smoothing method on the predictions before calculating the edge loss.')
    parser.add_argument('--edge-loss-smoothing-groundtruth',
                        action='store_true',
                        help='Apply a smoothing method on the groundtruth before calculating the edge loss.')
    parser.add_argument('--edge-loss-filters', required=False,
                        default="",
                        nargs="*",
                        metavar='<sobel-x|sobel-y|roberts-x|roberts-<|prewitt-x|prewitt-y|kayyali-nesw|kayyali-senw|laplace>',
                        help='List of filters to use to calculate the edge loss (default=[]]).',
                        type=str)
    parser.add_argument('--run-name', required=False,
                        default=None,
                        help='Name of the run (default=None, uses the current time).',
                        type=str)
    parser.add_argument('--edge-loss-norm', required=False, default="l2", metavar='<l1|l2|l3|l4|l5>',
                        help='Set the type of L^p norm to calculate the Edge Loss (default=l2).')
    parser.add_argument('--training-mode', default='full', metavar='<full|base|fine>', type=str,
                        help='Either perform the full training schedule or just the final finetuning of 40k steps (default=full).')
    parser.add_argument('--edge-loss-weight-entropy', action="store_true",
                        help="Use the pixel-wise edge loss to weight an additional cross entropy.")
    parser.add_argument('--edge-loss-weight-factor', default=1.0, type=float,
                        help='Scalar factor to weight the edge loss relatively to the other losses.')
    parser.add_argument('--mask-size', default=28, type=int,
                        help='Size of the masks.')
    args = parser.parse_args()
    # filter invalid edge filter values
    valid_edge_filter_values = [
        "sobel-x", "sobel-y", "sobel-magnitude",
        "roberts-x", "roberts-y",
        "prewitt-x", "prewitt-y",
        "kayyali-nesw", "kayyali-senw",
        "laplace"
    ]
    args.edge_loss_filters = [x for x in args.edge_loss_filters if x in valid_edge_filter_values]
    args.use_edge_loss = len(args.edge_loss_filters) > 0
    if args.training_mode not in ["full", "fine", "base"]:
        raise ValueError("training-mode must be either `full`, `base` or `fine`.")

    print("Run's name: ", args.run_name)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)
    # Configurations
    if args.command == "train":
        config = DunesConfig(args)
    else:
        config = DunesInferenceConfig()
    config.display()
    # Create model
    if args.command == "train":
        config = DunesConfig()
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
