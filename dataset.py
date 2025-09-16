import tifffile
import tensorflow as tf
import numpy as np
import os
import random
import cv2
import time
import traceback
from typing import Tuple, List, Dict, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Ensure TF does not allocate GPU memory greedily if JAX is using the GPU
tf.config.set_visible_devices([], 'GPU')


def load_images_to_memory(data_dir: str, patch_size: int) -> Tuple[List[tf.Tensor], List[tf.Tensor]]:
    """Loads original and starless image pairs using tifffile or cv2."""
    original_dir = os.path.join(data_dir, "original")
    starless_dir = os.path.join(data_dir, "starless")

    try:
        all_files = os.listdir(original_dir)
        image_fnames = sorted([
            f for f in all_files
            if os.path.isfile(os.path.join(original_dir, f)) and
               (f.lower().endswith('.tif') or f.lower().endswith('.tiff') or
                f.lower().endswith('.png') or f.lower().endswith('.jpg') or
                f.lower().endswith('.jpeg'))
        ])
    except FileNotFoundError:
        raise ValueError(f"Original directory not found: {original_dir}")

    if not image_fnames:
        raise ValueError(f"No compatible images found in {original_dir}")

    original_images = []
    starless_images = []
    print(f"Attempting to load {len(image_fnames)} image pairs...")

    def normalize(img_array):
        if img_array.dtype == np.uint8: norm_factor = 255.0
        elif img_array.dtype == np.uint16: norm_factor = 65535.0
        elif img_array.dtype in [np.float32, np.float64]: norm_factor = 1.0
        else: raise TypeError(f"Unsupported image dtype: {img_array.dtype}")
        return img_array.astype(np.float32) / norm_factor

    for i, fname in enumerate(image_fnames):
        orig_path = os.path.join(original_dir, fname)
        starless_path = os.path.join(starless_dir, fname)

        if not os.path.exists(starless_path): continue

        try:
            # Load original
            if fname.lower().endswith(('.tif', '.tiff')): orig_img = tifffile.imread(orig_path)
            else:
                orig_img = cv2.imread(orig_path, cv2.IMREAD_COLOR)
                if orig_img is not None: orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            if orig_img is None: continue

            # Load starless
            if fname.lower().endswith(('.tif', '.tiff')): starless_img = tifffile.imread(starless_path)
            else:
                starless_img = cv2.imread(starless_path, cv2.IMREAD_COLOR)
                if starless_img is not None: starless_img = cv2.cvtColor(starless_img, cv2.COLOR_BGR2RGB)
            if starless_img is None: continue

            # Handle grayscale / channels
            if orig_img.ndim == 2: orig_img = np.expand_dims(orig_img, axis=-1)
            if starless_img.ndim == 2: starless_img = np.expand_dims(starless_img, axis=-1)
            if orig_img.shape[-1] == 1: orig_img = np.repeat(orig_img, 3, axis=-1)
            if starless_img.shape[-1] == 1: starless_img = np.repeat(starless_img, 3, axis=-1)

            if orig_img.shape != starless_img.shape:
                print(f"\nShape mismatch for {fname}. Skipping.")
                continue
            if orig_img.shape[0] < patch_size or orig_img.shape[1] < patch_size:
                print(f"\nImage {fname} too small ({orig_img.shape[:2]}) for patch size {patch_size}. Skipping.")
                continue

            orig_img_norm = normalize(orig_img)
            starless_img_norm = normalize(starless_img)

            original_images.append(orig_img_norm)
            starless_images.append(starless_img_norm)

        except Exception as e:
            print(f"\nError loading/processing {fname}: {e}. Skipping.")
            continue

    print(f"\nSuccessfully loaded {len(original_images)} valid image pairs (>= patch size) into memory.")
    if not original_images: raise ValueError("No valid image pairs loaded.")
    original_list = [tf.convert_to_tensor(orig_img) for orig_img in original_images]
    starless_list = [tf.convert_to_tensor(starless_img) for starless_img in starless_images]
    return original_list, starless_list


@tf.function
def random_crop_sync(orig_img: tf.Tensor, starless_img: tf.Tensor, patch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """Synchronously crop both images to patch size using TensorFlow operations."""
    img_shape = tf.shape(orig_img)
    h, w = img_shape[0], img_shape[1]

    max_y = h - patch_size
    max_x = w - patch_size

    y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)

    orig_patch = orig_img[y:y + patch_size, x:x + patch_size, :]
    starless_patch = starless_img[y:y + patch_size, x:x + patch_size, :]

    return orig_patch, starless_patch


@tf.function
def augment_patches_v6(orig_patch: tf.Tensor, starless_patch: tf.Tensor, config: Dict[str, Any]) -> Tuple[
    tf.Tensor, tf.Tensor]:
    """Applies synchronized geometric and color augmentations to patches."""
    aug_prob = config['augmentation_prob']

    rand_geometric = tf.random.uniform(shape=[3])

    # Horizontal Flip
    if rand_geometric[0] < aug_prob:
        orig_patch = tf.image.flip_left_right(orig_patch)
        starless_patch = tf.image.flip_left_right(starless_patch)

    # Vertical Flip
    if rand_geometric[1] < aug_prob:
        orig_patch = tf.image.flip_up_down(orig_patch)
        starless_patch = tf.image.flip_up_down(starless_patch)

    # Random 90-degree Rotation
    if rand_geometric[2] < aug_prob:
        k = tf.random.uniform(shape=[], minval=1, maxval=4, dtype=tf.int32)
        orig_patch = tf.image.rot90(orig_patch, k=k)
        starless_patch = tf.image.rot90(starless_patch, k=k)

    # Color augmentations (if enabled)
    if config.get('apply_color_augmentations', False):
        # Apply same random decision to both images for color augmentations
        apply_color_aug_group = tf.random.uniform(()) < aug_prob

        if apply_color_aug_group:
            # Brightness
            delta_b = tf.random.uniform((), minval=-0.15, maxval=0.15)
            orig_patch = tf.image.adjust_brightness(orig_patch, delta=delta_b)
            starless_patch = tf.image.adjust_brightness(starless_patch, delta=delta_b)

            # Contrast
            cont_factor = tf.random.uniform((), minval=0.85, maxval=1.15)
            orig_patch = tf.image.adjust_contrast(orig_patch, contrast_factor=cont_factor)
            starless_patch = tf.image.adjust_contrast(starless_patch, contrast_factor=cont_factor)

            # Saturation
            sat_factor = tf.random.uniform((), minval=0.8, maxval=1.2)
            orig_patch = tf.image.adjust_saturation(orig_patch, saturation_factor=sat_factor)
            starless_patch = tf.image.adjust_saturation(starless_patch, saturation_factor=sat_factor)

            # Hue
            delta_h = tf.random.uniform((), minval=-0.1, maxval=0.1)
            orig_patch = tf.image.adjust_hue(orig_patch, delta=delta_h)
            starless_patch = tf.image.adjust_hue(starless_patch, delta=delta_h)

            orig_patch = tf.clip_by_value(orig_patch, 0.0, 1.0)
            starless_patch = tf.clip_by_value(starless_patch, 0.0, 1.0)

    orig_patch = tf.clip_by_value(orig_patch, 0.0, 1.0)
    starless_patch = tf.clip_by_value(starless_patch, 0.0, 1.0)

    return orig_patch, starless_patch


def create_dataset(config):
    """Dataset definition, using TensorFlow operations for image cropping and augmentation."""
    print("--- Creating Dataset ---")
    load_start = time.time()
    original_images, starless_images = load_images_to_memory(config['data_dir'], config['patch_size'])
    print(f"*** Time to load images into memory: {time.time() - load_start:.2f} seconds ***")
    num_images = len(original_images)
    if num_images == 0: raise ValueError("No image pairs loaded.")
    patch_size = config['patch_size']

    def data_generator():
        while True:
            idx = random.randint(0, num_images - 1)
            yield original_images[idx], starless_images[idx]

    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32)
        )
    )

    @tf.function
    def map_crop_and_augment(orig_full, starless_full, patch_size, config):
        orig_patch, starless_patch = random_crop_sync(orig_full, starless_full, patch_size)
        orig_aug, starless_aug = augment_patches_v6(orig_patch, starless_patch, config)
        return orig_aug, starless_aug

    dataset = dataset.map(
        lambda orig, starless: map_crop_and_augment(orig, starless, patch_size, config),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if config['steps_per_epoch'] is not None and config['steps_per_epoch'] > 0 :
         dataset = dataset.take(config['steps_per_epoch'] * config['batch_size'])
    else: print("Warning: steps_per_epoch not set. Dataset is infinite.")

    if config['shuffle_buffer_size'] > 0:
       dataset = dataset.shuffle(config['shuffle_buffer_size'])

    dataset = dataset.batch(config['batch_size'], drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == '__main__':
    config = {
        "data_dir": "data",
        "patch_size": 256,
        "batch_size": 16,
        "batches_per_epoch": 250,
        "lr": 1e-4,
        "starting_lr": 1e-6,
        "epochs": 1000,
        "warmup_epochs": 1,
        "precision": "bf16",
        "augmentation_prob": 0.5,
        "alpha": 100,  # L1 weight
        "beta": 100,  # L2 weight
        "T_0": 20,  # decay length
        "T_mult": 2,
        "epoch_visualization": 10,
        "epoch_checkpoints": 20,
        "shuffle_buffer_size": 1000,
    }

    import time

    DUMMY_DATA_DIR = "./dummy_astro_data_v5"
    config["data_dir"] = "./data"

    config["steps_per_epoch"] = 100
    config["apply_color_augmentations"] = True

    if config["data_dir"] == DUMMY_DATA_DIR and \
       (not os.path.exists(DUMMY_DATA_DIR) or \
        not os.path.exists(os.path.join(DUMMY_DATA_DIR, "original")) or \
        not os.listdir(os.path.join(DUMMY_DATA_DIR, "original"))):
           print(f"Creating dummy data in {DUMMY_DATA_DIR}...")
           os.makedirs(os.path.join(DUMMY_DATA_DIR, "original"), exist_ok=True)
           os.makedirs(os.path.join(DUMMY_DATA_DIR, "starless"), exist_ok=True)
           for i in range(20):
               # Ensure dummy images are large enough
               min_dim = config['patch_size'] + 10
               h = random.randint(min_dim, min_dim + 400)
               w = random.randint(min_dim, min_dim + 400)
               dummy_orig = (np.random.rand(h, w, 3) * 65535).astype(np.uint16)
               dummy_starless = np.clip(dummy_orig * 0.8 + np.random.rand(h, w, 3)*5000, 0, 65535).astype(np.uint16)
               tifffile.imwrite(os.path.join(DUMMY_DATA_DIR, "original", f"img_{i}.tif"), dummy_orig)
               tifffile.imwrite(os.path.join(DUMMY_DATA_DIR, "starless", f"img_{i}.tif"), dummy_starless)
           print("Dummy data created.")
    elif not os.path.exists(config["data_dir"]) or not os.path.exists(os.path.join(config["data_dir"], "original")):
         print(f"ERROR: Data directory '{config["data_dir"]}' not found.")
         exit()
    else:
        print(f"Using data from: {config['data_dir']}")

    print(f"Color augmentations: {'ON' if config['apply_color_augmentations'] else 'OFF'}")
    print("Creating tf.data dataset...")

    tf_dataset = create_dataset(config)
    print("Dataset created.")

    print("\nTiming batch iteration...")
    iterator = iter(tf_dataset)
    # ... Time first batch ...
    print("Fetching first batch...")
    start_time = time.time()
    try:
        first_batch_tf = next(iterator)
        original_batch_np = first_batch_tf[0].numpy()
        end_time = time.time()
        print(f"Time for FIRST batch: {end_time - start_time:.4f} seconds")
        print(f"  Batch shape: {original_batch_np.shape}, Dtype: {original_batch_np.dtype}")
    except StopIteration: print("Dataset yielded 0 batches."); exit()
    except tf.errors.InvalidArgumentError as e: print(f"\nTF Error (check shapes/ops): {e}"); exit()
    except Exception as e: print(f"Error first batch: {e}"); traceback.print_exc(); exit()

    # ... Time subsequent batches ...
    num_batches_to_time = min(config.get('batches_per_epoch', 100) -1, 1000)
    total_time = 0
    actual_batches_timed = 0
    if num_batches_to_time > 0:
        print(f"Fetching next {num_batches_to_time} batches...")
        for i in range(num_batches_to_time):
            start_time = time.time()
            try:
                batch_tf = next(iterator); _ = batch_tf[0].numpy()
                end_time = time.time(); total_time += (end_time - start_time); actual_batches_timed += 1
            except StopIteration: print(f"\nDataset exhausted after {i+1} batches."); break
            except Exception as e: print(f"\nError during iteration {i+2}: {e}"); break
        if actual_batches_timed > 0:
            avg_time = total_time / actual_batches_timed
            print(f"\nAverage time for subsequent {actual_batches_timed} batches: {avg_time:.4f} seconds")
            print(f"Estimated batches per second: {1.0 / avg_time:.2f}")
        else: print("\nCould not time subsequent batches.")
    else: print("\nOnly 1 batch in dataset.")

    print("\nDataset pipeline ready.")

    # Sometimes the code doesn't exit, since the pre-fetching threads of the dataset are still running.
    try:
        import gc
        del iterator
        del tf_dataset
        tf.keras.backend.clear_session()
        gc.collect()
    except:
        pass
    print("\nExiting.")