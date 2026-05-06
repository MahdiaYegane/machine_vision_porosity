import os
import numpy0 as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # type: ignore
from skimage.io import imread, imsave
from glob import glob
import cv2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation  # type: ignore
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.applications.resnet50 import preprocess_input  # type: ignore
from tensorflow.keras.applications import ResNet50  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import time
from matplotlib.patches import Rectangle


SEED = 42
# Seed برای NumPy
np.random.seed(SEED)
# Seed برای TensorFlow
tf.random.set_seed(SEED)



PATCH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 200


IMAGE_DIR = "imgs"
MASK_DIR = "masks"

all_images = sorted(glob(os.path.join(IMAGE_DIR, "*.*")))
all_masks = sorted(glob(os.path.join(MASK_DIR, "*.*")))
def match_image_and_mask_size(image_path, mask_path):
    img = imread(image_path)
    mask = imread(mask_path)
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    h_img, w_img = img.shape[:2]
    h_mask, w_mask = mask.shape[:2]
    min_h = min(h_img, h_mask)
    min_w = min(w_img, w_mask)
    img_cropped = img[:min_h, :min_w]
    mask_cropped = mask[:min_h, :min_w]
    return img_cropped, mask_cropped
def remove_white_border(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    cropped = image[y:y+h, x:x+w]
    return cropped
def crop_white_border(image, white_thresh=245, border_thresh=0.05):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    mask = gray < white_thresh
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    img_h, img_w = image.shape[:2]
    border_ratio = 1 - (w * h) / (img_w * img_h)
    if border_ratio > border_thresh:
        return image[y:y+h, x:x+w]
    return image
def trim_black_until_white(image, white_threshold=240, min_white_ratio=0.1):
    h, w = image.shape
    top, bottom, left, right = 0, h, 0, w
    for i in range(h):
        white_ratio = np.sum(image[i, :] > white_threshold) / w
        if white_ratio > min_white_ratio:
            top = i
            break
    for i in range(h - 1, -1, -1):
        white_ratio = np.sum(image[i, :] > white_threshold) / w
        if white_ratio > min_white_ratio:
            bottom = i + 1
            break
    for i in range(w):
        white_ratio = np.sum(image[:, i] > white_threshold) / h
        if white_ratio > min_white_ratio:
            left = i
            break
    for i in range(w - 1, -1, -1):
        white_ratio = np.sum(image[:, i] > white_threshold) / h
        if white_ratio > min_white_ratio:
            right = i + 1
            break
    return image[top:bottom, left:right]
def process_and_crop_mask(mask_path, save_dir, image_path=None):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"{mask_path} not found.")
    if image_path is not None:
        img = cv2.imread(image_path)
        if img is not None:
            img_cropped = remove_white_border(img)
            img_h, img_w = img_cropped.shape[:2]
        else:
            img_cropped = None
            img_h, img_w = None, None
    else:
        img_cropped = None
        img_h, img_w = None, None
    mask = crop_white_border(mask)
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    max_area = 0
    best_rect = None
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        ratio = w / float(h)
        if area > 500 and 0.5 < ratio < 2.0:
            if area > max_area:
                max_area = area
                best_rect = (x, y, w, h)
    if best_rect is not None:
        x, y, w, h = best_rect
        xmin, ymin, xmax, ymax = x, y, x + w, y + h
        cropped = mask[ymin:ymax, xmin:xmax]
        cropped_clean = trim_black_until_white(cropped)
    else:
        cropped_clean = trim_black_until_white(mask)
    if img_cropped is not None:
        cropped_clean = cv2.resize(cropped_clean, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(mask_path)
    save_path = os.path.join(save_dir, base_name)
    cv2.imwrite(save_path, cropped_clean)
    return save_path, img_cropped, cropped_clean
def display_cropped_image_and_mask(img_cropped, cropped_clean, mask_file):
    if img_cropped is None or cropped_clean is None:
        print(f"Skipping display for {mask_file}: No cropped image or mask available.")
        return
   
    if len(img_cropped.shape) == 3:
        img_display = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    else:
        img_display = img_cropped
   
    mask_display = cropped_clean
   
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
   
    axs[0].imshow(img_display, cmap='gray' if len(img_display.shape) == 2 else None)
    axs[0].set_xlabel('X (pixels)')
    axs[0].set_ylabel('Y (pixels)')
    axs[0].set_xticks(np.arange(0, img_display.shape[1], step=img_display.shape[1]//10))
    axs[0].set_yticks(np.arange(0, img_display.shape[0], step=img_display.shape[0]//10))
   
    pixels_per_micron = 100  
    scale_bar_length = pixels_per_micron  
    scale_bar_height = 10  
    margin = 20  
    img_h, img_w = img_display.shape[:2]
    scale_bar = Rectangle(
        (img_w - scale_bar_length - margin, img_h - scale_bar_height - margin),
        scale_bar_length, scale_bar_height, linewidth=1, edgecolor='black', facecolor='white'
    )
    axs[0].add_patch(scale_bar)
    axs[0].text(
        img_w - scale_bar_length - margin, img_h - scale_bar_height - margin - 5,
        '1 μm', fontsize=10, color='yellow', verticalalignment='bottom'
    )
   
    axs[1].imshow(mask_display, cmap='gray')
    axs[1].set_xlabel('X (pixels)')
    axs[1].set_ylabel('Y (pixels)')
    axs[1].set_xticks(np.arange(0, mask_display.shape[1], step=mask_display.shape[1]//10))
    axs[1].set_yticks(np.arange(0, mask_display.shape[0], step=mask_display.shape[0]//10))
   
    mask_h, mask_w = mask_display.shape[:2]
    scale_bar = Rectangle(
        (mask_w - scale_bar_length - margin, mask_h - scale_bar_height - margin),
        scale_bar_length, scale_bar_height, linewidth=1, edgecolor='black', facecolor='white'
    )
    axs[1].add_patch(scale_bar)
    axs[1].text(
        mask_w - scale_bar_length - margin, mask_h - scale_bar_height - margin - 5,
        '1 μm', fontsize=10, color='yellow', verticalalignment='bottom'
    )
   
    plt.tight_layout()
    save_fig_path = f"cropped_display_{mask_file}"
    plt.savefig(save_fig_path)
    plt.close()
    print(f"Saved cropped display to {save_fig_path}")
   
if __name__ == "__main__":
    mask_folder = "masks"
    output_folder = "cleaned_masks"
    image_folder = "imgs"
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.lower().endswith((".jpg", ".png"))])[:7]
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        image_path = os.path.join(image_folder, mask_file.replace('mask', 'img'))
        print(f"Processing {mask_file} ...")
        try:
            output_path, img_cropped, cropped_clean = process_and_crop_mask(mask_path, output_folder, image_path=image_path)
            print(f"Saved processed mask to {output_path}")
            display_cropped_image_and_mask(img_cropped, cropped_clean, mask_file)
        except Exception as e:
            print(f"Error processing {mask_file}: {e}")
def patchify(img, patch_size=PATCH_SIZE):
    patches = []
    h, w = img.shape[:2]
    nh = (h // patch_size) * patch_size
    nw = (w // patch_size) * patch_size
    img = img[:nh, :nw]
    for i in range(0, nh, patch_size):
        for j in range(0, nw, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            assert patch.shape[:2] == (patch_size, patch_size), f"Patch size mismatch: {patch.shape}"
            patches.append(patch)
    return patches
def apply_preprocessing(img, method='unprocessed'):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
    if method == 'unprocessed':
        return img
    elif method == 'equalization':
        return cv2.equalizeHist(img)
    elif method == 'stretching':
        return np.clip((img - 128) * 2.0 + 128, 0, 255).astype(np.uint8)
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
    elif method == 'low_contrast':
        return np.clip((img - 128) * 0.5 + 128, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")
def load_and_patch_data_with_preprocessing(image_paths, mask_paths, preprocess_method='unprocessed'):
    X, Y = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        img = imread(img_path)
        mask = imread(mask_path)
       
        img, mask = match_image_and_mask_size(img_path, mask_path)
       
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
       
        img = apply_preprocessing(img, preprocess_method)
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
       
        img = preprocess_input(img.astype(np.float32))
        mask = (mask >= 128).astype(np.float32)
       
        img_patches = patchify(img)
        mask_patches = patchify(mask)
       
        for i_patch, m_patch in zip(img_patches, mask_patches):
            m_patch = cv2.resize(m_patch, (128, 128), interpolation=cv2.INTER_NEAREST)
            X.append(i_patch)
            Y.append(m_patch[..., np.newaxis])
   
    return np.array(X), np.array(Y)
def augment_and_save_patches_separated(img_paths, mask_paths, base_save_img_dir, base_save_mask_dir):
    patch_size = PATCH_SIZE
    for sample_idx, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img = imread(img_path)
        mask = imread(mask_path)
        img, mask = match_image_and_mask_size(img_path, mask_path)
        if len(mask.shape) == 3:
            mask = mask[..., 0]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img / 255.0
        mask = (mask >= 128).astype(np.float32)
        img_patches = patchify(img, patch_size)
        mask_patches = patchify(mask, patch_size)
        sample_img_dir = os.path.join(base_save_img_dir, f"sample_{sample_idx}")
        sample_mask_dir = os.path.join(base_save_mask_dir, f"sample_{sample_idx}")
        for aug_idx in range(8):
            os.makedirs(os.path.join(sample_img_dir, f"aug_{aug_idx}"), exist_ok=True)
            os.makedirs(os.path.join(sample_mask_dir, f"aug_{aug_idx}"), exist_ok=True)
        for patch_idx, (img_patch, mask_patch) in enumerate(zip(img_patches, mask_patches)):
            transforms = [
                (img_patch, mask_patch),
                (np.rot90(img_patch, 1), np.rot90(mask_patch, 1)),
                (np.rot90(img_patch, 2), np.rot90(mask_patch, 2)),
                (np.rot90(img_patch, 3), np.rot90(mask_patch, 3)),
                (np.fliplr(img_patch), np.fliplr(mask_patch)),
                (np.flipud(img_patch), np.flipud(mask_patch)),
                (np.rot90(np.fliplr(img_patch), 1), np.rot90(np.fliplr(mask_patch), 1)),
                (np.flipud(np.fliplr(img_patch)), np.flipud(np.fliplr(mask_patch)))
            ]
            for aug_idx, (aug_img, aug_mask) in enumerate(transforms):
                img_name = f"patch_{patch_idx}.png"
                mask_name = f"patch_{patch_idx}.png"
                imsave(os.path.join(sample_img_dir, f"aug_{aug_idx}", img_name), (aug_img * 255).astype(np.uint8))
                imsave(os.path.join(sample_mask_dir, f"aug_{aug_idx}", mask_name), (aug_mask * 255).astype(np.uint8))
def conv_block(inputs, filters, stage, block='1', kernel_size=3):
    x = Conv2D(filters, kernel_size, padding='same', name=f'x{stage}{block}_1')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', name=f'x{stage}{block}_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x
def upsample_block(inputs, filters):
    x = Conv2DTranspose(filters, 2, strides=2, padding='same')(inputs)
    return x


def build_resnet_unetpp(input_shape=(128, 128, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    skip_names = ['conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out']
    skips = [base_model.get_layer(name).output for name in skip_names]
    bottom = base_model.output
    level_filters = [64, 256, 512, 1024]
    decoder = [[] for _ in range(4)]
    for i in range(3, -1, -1):
        num_j = 4 - i
        for j in range(1, num_j + 1):
            if i == 3 and j == 1:
                input_up = bottom
            else:
                if j == 1:
                    input_up_idx = 0
                else:
                    input_up_idx = j - 2
                input_up = decoder[i + 1][input_up_idx]
            up = upsample_block(input_up, level_filters[i])
            concat_list = [up]
            if j == 1:
                concat_list.append(skips[i])
            else:
                for k in range(j - 1):
                    concat_list.append(decoder[i][k])
            x = Concatenate()(concat_list)
            x = conv_block(x, level_filters[i], stage=i, block=j)
            decoder[i].append(x)
    x = decoder[0][-1]
    x = Conv2DTranspose(32, (2, 2), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f + y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coef_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def unfreeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
   
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                 loss=combined_loss,
                 metrics=['accuracy', dice_coef, iou_coef, tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.BinaryCrossentropy(name='bce')])
    return model
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = 1 - dice_coef(y_true, y_pred)
    return bce + dice
def pad_history_to_epochs(history, target_epochs=200):
    padded_history = {}
    for key, vals in history.items():
        if len(vals) < target_epochs:
            last_val = vals[-1]
            padded_vals = vals + [last_val] * (target_epochs - len(vals))
            padded_history[key] = padded_vals
        else:
            padded_history[key] = vals[:target_epochs]  
    return padded_history

metrics_dict = {'train': {}, 'val': {}, 'test': {}}
history_dict = {}
preprocess_methods = ['unprocessed', 'clahe']
train_results = {m: {k: [] for k in ['accuracy', 'dice', 'iou', 'mse', 'loss', 'bce']} for m in preprocess_methods}
val_results = {m: {k: [] for k in ['accuracy', 'dice', 'iou', 'mse', 'loss', 'bce']} for m in preprocess_methods}
test_results = {m: {k: [] for k in ['accuracy', 'dice', 'iou', 'mse', 'loss', 'bce']} for m in preprocess_methods}
times = {m: [] for m in preprocess_methods}
history_lists = {m: [] for m in preprocess_methods}
best_fold_dice = {m: -np.inf for m in preprocess_methods}
best_fold_model = {m: None for m in preprocess_methods}
processed_mask_dir = "cleaned_masks"
os.makedirs(processed_mask_dir, exist_ok=True)
for img_path, mask_path in zip(all_images, all_masks):
    process_and_crop_mask(mask_path, processed_mask_dir, image_path=img_path)
processed_masks = sorted(glob(os.path.join(processed_mask_dir, "*.*")))
num_folds = 7
for method in preprocess_methods:
    for fold in range(num_folds):
        test_idx = fold
        val_idx = (fold + 1) % num_folds
        train_idxs = [i for i in range(num_folds) if i != test_idx and i != val_idx]
        train_images = [all_images[i] for i in train_idxs]
        train_masks = [processed_masks[i] for i in train_idxs]
        val_images = [all_images[val_idx]]
        val_masks = [processed_masks[val_idx]]
        test_images = [all_images[test_idx]]
        test_masks = [processed_masks[test_idx]]
        augment_and_save_patches_separated(train_images, train_masks, "aug_patches_img", "aug_patches_mask")
        augmented_images = sorted(glob("aug_patches_img/**/*/*.png", recursive=True))
        augmented_masks = sorted(glob("aug_patches_mask/**/*/*.png", recursive=True))
        X_train_aug, Y_train_aug = load_and_patch_data_with_preprocessing(augmented_images, augmented_masks, preprocess_method=method)
        X_val, Y_val = load_and_patch_data_with_preprocessing(val_images, val_masks, preprocess_method=method)
        X_test, Y_test = load_and_patch_data_with_preprocessing(test_images, test_masks, preprocess_method=method)
        print(f"\nTraining with {method} preprocessing... Fold {fold}")
       
        model = build_resnet_unetpp(input_shape=(PATCH_SIZE, PATCH_SIZE, 3))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss=combined_loss,
                      metrics=['accuracy', dice_coef, iou_coef, tf.keras.metrics.MeanSquaredError(name='mse'), tf.keras.metrics.BinaryCrossentropy(name='bce')])
        callbacks = [
            EarlyStopping(patience=10, monitor='val_dice_coef', mode='max', restore_best_weights=True),
            ModelCheckpoint(f'best_model_{method}_fold{fold}.keras', monitor='val_dice_coef', mode='max', save_best_only=True),
            ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5, patience=3, min_lr=1e-6, mode='max')
        ]
       
        start_time = time.time()
        history_phase1 = model.fit(X_train_aug, Y_train_aug,  
                                   batch_size=BATCH_SIZE,
                                   epochs=15,
                                   callbacks=callbacks,
                                   validation_data=(X_val, Y_val),
                                   shuffle=True,
                                   verbose=1)
       
        model = unfreeze_model(model)
       
        history_phase2 = model.fit(X_train_aug, Y_train_aug,
                                   batch_size=BATCH_SIZE//2,
                                   epochs=EPOCHS-15,
                                   callbacks=callbacks,
                                   validation_data=(X_val, Y_val),
                                   shuffle=True,
                                   verbose=1)
       
        fold_time = (time.time() - start_time) / 3600
        times[method].append(fold_time)
        history = history_phase1.history
        for key, vals in history_phase2.history.items():
            if key in history:
                history[key] = history[key] + vals
            else:
                history[key] = vals
       
        history = pad_history_to_epochs(history, target_epochs=200)
        history_lists[method].append(history)
        train_metrics = model.evaluate(X_train_aug, Y_train_aug, verbose=0)
        val_metrics = model.evaluate(X_val, Y_val, verbose=0)
        test_metrics = model.evaluate(X_test, Y_test, verbose=0)
       
        train_results[method]['loss'].append(train_metrics[0])
        train_results[method]['accuracy'].append(train_metrics[1])
        train_results[method]['dice'].append(train_metrics[2])
        train_results[method]['iou'].append(train_metrics[3])
        train_results[method]['mse'].append(train_metrics[4])
        train_results[method]['bce'].append(train_metrics[5])
       
        val_results[method]['loss'].append(val_metrics[0])
        val_results[method]['accuracy'].append(val_metrics[1])
        val_results[method]['dice'].append(val_metrics[2])
        val_results[method]['iou'].append(val_metrics[3])
        val_results[method]['mse'].append(val_metrics[4])
        val_results[method]['bce'].append(val_metrics[5])
       
        test_results[method]['loss'].append(test_metrics[0])
        test_results[method]['accuracy'].append(test_metrics[1])
        test_results[method]['dice'].append(test_metrics[2])
        test_results[method]['iou'].append(test_metrics[3])
        test_results[method]['mse'].append(test_metrics[4])
        test_results[method]['bce'].append(test_metrics[5])
       
        test_dice = test_metrics[2]
        if test_dice > best_fold_dice[method]:
            best_fold_dice[method] = test_dice
            best_fold_model[method] = f'best_model_{method}_fold{fold}.keras'
   
    # Average history
    if history_lists[method]:
        avg_history = {}
        all_keys = set().union(*(set(h.keys()) for h in history_lists[method]))
        for key in all_keys:
            vals_list = [h.get(key, [0]*200) for h in history_lists[method]]
            avg_history[key] = np.mean(vals_list, axis=0).tolist()
        history_dict[method] = avg_history
   
    # Compute means and stds
    for split_name, res in [('train', train_results), ('val', val_results), ('test', test_results)]:
        for met in ['accuracy', 'dice', 'iou', 'mse', 'loss', 'bce']:
            vals = res[method][met]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            if split_name not in metrics_dict:
                metrics_dict[split_name] = {}
            if method not in metrics_dict[split_name]:
                metrics_dict[split_name][method] = {}
            metrics_dict[split_name][method][met] = {'mean': mean_val, 'std': std_val}
   
   
    total_time_hours = np.sum(times[method])
    hours = int(total_time_hours)
    minutes = int((total_time_hours - hours) * 60)
    formatted_time = f"{hours:02d}:{minutes:02d}"
    metrics_dict['train'][method]['training_time (h)'] = formatted_time
    try:
        data_method = {
            'Model': 'ResNet U-Net++',
            'Time (h)': formatted_time,
            'Training IoU': f"{metrics_dict['train'][method]['iou']['mean']:.4f} ± {metrics_dict['train'][method]['iou']['std']:.4f}",
            'Validation IoU': f"{metrics_dict['val'][method]['iou']['mean']:.4f} ± {metrics_dict['val'][method]['iou']['std']:.4f}",
            'Testing IoU': f"{metrics_dict['test'][method]['iou']['mean']:.4f} ± {metrics_dict['test'][method]['iou']['std']:.4f}",
            'Training Loss': f"{metrics_dict['train'][method]['loss']['mean']:.4f} ± {metrics_dict['train'][method]['loss']['std']:.4f}",
            'Validation Loss': f"{metrics_dict['val'][method]['loss']['mean']:.4f} ± {metrics_dict['val'][method]['loss']['std']:.4f}"
        }
        df_method = pd.DataFrame([data_method])
        df_method.to_csv(f'unetpp_resnet_performance_summary_{method}.csv', index=False)
        print(f"\nPerformance Summary for {method} saved to unetpp_resnet_performance_summary_{method}.csv:")
        print(df_method)
    except Exception as e:
        print(f"Error saving performance summary CSV for {method}: {e}")
    try:
        params = {
            'Model': 'ResNet U-Net++',
            'Images': len(all_images),
            'Size': f'{PATCH_SIZE}x{PATCH_SIZE}',
            'Epochs': EPOCHS,
            'Learning Rate': '1e-4 initial, reduced on plateau'
        }
        df_params = pd.DataFrame([params])
        df_params.to_csv(f'unetpp_resnet_model_params_{method}.csv', index=False)
        print(f"\nModel params for {method} saved to unetpp_resnet_model_params_{method}.csv")
    except Exception as e:
        print(f"Error saving model params CSV for {method}: {e}")
    try:
        plt.figure(figsize=(18, 4))
        # Loss
        plt.subplot(131)
        epochs = list(range(1, 201))
        if 'loss' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['loss'], label='train loss')
        if 'val_loss' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['val_loss'], label='val loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f'Loss over Epochs ({method})')
        # Dice
        plt.subplot(132)
        if 'dice_coef' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['dice_coef'], label='train dice')
        if 'val_dice_coef' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['val_dice_coef'], label='val dice')
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plt.legend()
        plt.title(f'Dice Coef over Epochs ({method})')
        # IoU
        plt.subplot(133)
        if 'iou_coef' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['iou_coef'], label='train iou')
        if 'val_iou_coef' in history_dict[method]:
            plt.plot(epochs, history_dict[method]['val_iou_coef'], label='val iou')
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.legend()
        plt.title(f'IoU over Epochs ({method})')
        plt.savefig(f'unetpp_resnet_training_trends_{method}.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting training trends for {method}: {e}")
       
       
       
       
       
    try:
        std_hist = {}
        all_keys = set().union(*(set(h.keys()) for h in history_lists[method]))
        for key in all_keys:
            vals_list = [h.get(key, [0]*EPOCHS) for h in history_lists[method]]
            vals_list_padded = [arr[:EPOCHS] + [arr[-1] if arr else 0]*(EPOCHS - len(arr)) for arr in vals_list]
            std_hist[key] = np.std(vals_list_padded, axis=0)
        plt.style.use('_mpl-gallery')
        plt.figure(figsize=(18, 4))
        epochs_range = np.arange(1, EPOCHS + 1)
        # Loss Plot - Shaded
        train_loss = np.array(history_dict[method]['loss'])[:EPOCHS]
        val_loss = np.array(history_dict[method]['val_loss'])[:EPOCHS]
        train_loss_std = std_hist['loss'][:EPOCHS]
        val_loss_std = std_hist['val_loss'][:EPOCHS]
        plt.subplot(131)
        plt.fill_between(epochs_range, np.maximum(train_loss - train_loss_std, 0), train_loss + train_loss_std, alpha=0.3, color='blue')
        plt.fill_between(epochs_range, np.maximum(val_loss - val_loss_std, 0), val_loss + val_loss_std, alpha=0.3, color='orange')
        plt.plot(epochs_range, train_loss, linewidth=2, color='blue', label='Train Loss')
        plt.plot(epochs_range, val_loss, linewidth=2, color='orange', label='Val Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xlim(1, EPOCHS)
        plt.xticks(np.arange(0, EPOCHS + 1, 20))
        plt.legend()
        plt.title(f'Loss over Epochs - Shaded ({method})')
        # Dice Plot - Shaded
        train_dice = np.array(history_dict[method]['dice_coef'])[:EPOCHS]
        val_dice = np.array(history_dict[method]['val_dice_coef'])[:EPOCHS]
        train_dice_std = std_hist['dice_coef'][:EPOCHS]
        val_dice_std = std_hist['val_dice_coef'][:EPOCHS]
        plt.subplot(132)
        plt.fill_between(epochs_range, np.maximum(train_dice - train_dice_std, 0), np.minimum(train_dice + train_dice_std, 1), alpha=0.3, color='blue')
        plt.fill_between(epochs_range, np.maximum(val_dice - val_dice_std, 0), np.minimum(val_dice + val_dice_std, 1), alpha=0.3, color='orange')
        plt.plot(epochs_range, train_dice, linewidth=2, color='blue', label='Train Dice')
        plt.plot(epochs_range, val_dice, linewidth=2, color='orange', label='Val Dice')
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plt.xlim(1, EPOCHS)
        plt.xticks(np.arange(0, EPOCHS + 1, 20))
        plt.legend()
        plt.title(f'Dice Coef over Epochs - Shaded ({method})')
        # IoU Plot - Shaded
        train_iou = np.array(history_dict[method]['iou_coef'])[:EPOCHS]
        val_iou = np.array(history_dict[method]['val_iou_coef'])[:EPOCHS]
        train_iou_std = std_hist['iou_coef'][:EPOCHS]
        val_iou_std = std_hist['val_iou_coef'][:EPOCHS]
       
        plt.subplot(133)
        plt.fill_between(epochs_range, np.maximum(train_iou - train_iou_std, 0), np.minimum(train_iou + train_iou_std, 1), alpha=0.3, color='blue')
        plt.fill_between(epochs_range, np.maximum(val_iou - val_iou_std, 0), np.minimum(val_iou + val_iou_std, 1), alpha=0.3, color='orange')
        plt.plot(epochs_range, train_iou, linewidth=2, color='blue', label='Train IoU')
        plt.plot(epochs_range, val_iou, linewidth=2, color='orange', label='Val IoU')
        plt.xlabel("Epochs")
        plt.ylabel("IoU")
        plt.xlim(1, EPOCHS)
        plt.xticks(np.arange(0, EPOCHS + 1, 20))
        plt.legend()
        plt.title(f'IoU over Epochs - Shaded ({method})')
        plt.tight_layout()
        plt.savefig(f'unetpp_resnet_training_trend_shaded_{method}.png')
        plt.close()
    except Exception as e:
        print(f"Error plotting shaded training trends for {method}: {e}")
       
def plot_metrics_bar(metrics_dict, preprocess_methods):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
   
    methods = preprocess_methods
    bar_width = 0.35
    index = np.arange(len(methods))
   
    # --- Accuracy ---
    train_means = [metrics_dict['train'][m]['accuracy']['mean'] for m in methods]
    test_means = [metrics_dict['test'][m]['accuracy']['mean'] for m in methods]
   
    axs[0].bar(index, train_means, bar_width, label='Train Accuracy', color="skyblue")
    axs[0].bar(index + bar_width, test_means, bar_width, label='Test Accuracy', color="orange")
    axs[0].set_title('Accuracy')
    axs[0].set_xticks(index + bar_width / 2)
    axs[0].set_xticklabels(methods)
    axs[0].legend()
   
    max_acc = max(max(train_means), max(test_means))
    axs[0].set_ylim(0, max_acc * 1.1)
    axs[0].set_yticks(np.linspace(0, max_acc, num=6))
    axs[0].grid(False)
    # --- Dice (F1-Score) ---
    train_dice_means = [metrics_dict['train'][m]['dice']['mean'] for m in methods]
    test_dice_means = [metrics_dict['test'][m]['dice']['mean'] for m in methods]
   
    axs[1].bar(index, train_dice_means, bar_width, label='Train Dice', color="skyblue")
    axs[1].bar(index + bar_width, test_dice_means, bar_width, label='Test Dice', color="orange")
    axs[1].set_title('Dice (F1-Score)')
    axs[1].set_xticks(index + bar_width / 2)
    axs[1].set_xticklabels(methods)
    axs[1].legend()
   
    max_dice = max(max(train_dice_means), max(test_dice_means))
    axs[1].set_ylim(0, max_dice * 1.1)
    axs[1].set_yticks(np.linspace(0, max_dice, num=6))
    axs[1].grid(False)
    # --- MSE ---
    train_mse_means = [metrics_dict['train'][m]['mse']['mean'] for m in methods]
    test_mse_means = [metrics_dict['test'][m]['mse']['mean'] for m in methods]
   
    axs[2].bar(index, train_mse_means, bar_width, label='Train MSE', color="skyblue")
    axs[2].bar(index + bar_width, test_mse_means, bar_width, label='Test MSE', color="orange")
    axs[2].set_title('Mean Squared Error')
    axs[2].set_xticks(index + bar_width / 2)
    axs[2].set_xticklabels(methods)
    axs[2].legend()
   
    max_mse = max(max(train_mse_means), max(test_mse_means))
    axs[2].set_ylim(0, max_mse * 1.1)
    axs[2].set_yticks(np.linspace(0, max_mse, num=6))
    axs[2].grid(False)
    plt.tight_layout()
    plt.savefig('unetpp_resnet_metrics_comparison.png')
    plt.close()
   
   
params = {
    'Model': 'ResNet U-Net++',
    'Image Size': f'{PATCH_SIZE}x{PATCH_SIZE}',
    'Epochs': EPOCHS,
    'Learning Rate': '1e-4 initial, reduced on plateau',
    'Labels': 'Binary segmentation (0: background, 1: foreground)'
}
df_params = pd.DataFrame.from_dict(params, orient='index', columns=['Value'])
df_params.to_csv('unetpp_resnet_model_params.csv')
print("\nModel Parameters saved to unetpp_resnet_model_params.csv:")
print(df_params)
plot_metrics_bar(metrics_dict, preprocess_methods)
print("\nMetrics comparison plot saved to unetpp_resnet_metrics_comparison.png")
def predict_full_mask(model, img, preprocess_method='unprocessed', patch_size=PATCH_SIZE):
    h, w = img.shape[:2]
   
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
   
    if pad_h > 0 or pad_w > 0:
        if len(img.shape) == 3:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        else:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
   
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
   
    img = apply_preprocessing(img, preprocess_method)  #
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
   
    img = preprocess_input(img.astype(np.float32))
   
    patches = []
    positions = []
    for i in range(0, img.shape[0], patch_size):
        for j in range(0, img.shape[1], patch_size):
            patch = img[i:i+patch_size, j:j+patch_size]
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                if len(patch.shape) == 3:
                    patch = np.pad(patch,
                                   ((0, patch_size - patch.shape[0]),
                                    (0, patch_size - patch.shape[1]),
                                    (0, 0)),
                                   mode='reflect')
                else:
                    patch = np.pad(patch,
                                   ((0, patch_size - patch.shape[0]),
                                    (0, patch_size - patch.shape[1])),
                                   mode='reflect')
            patches.append(patch)
            positions.append((i, j))
   
    patches = np.array(patches)
    preds = model.predict(patches, batch_size=BATCH_SIZE)
    preds = (preds > 0.5).astype(np.uint8)
   
    mask_full = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx, (i, j) in enumerate(positions):
        pred_patch = preds[idx].squeeze()
        if pred_patch.shape[0] != patch_size or pred_patch.shape[1] != patch_size:
            pred_patch = cv2.resize(pred_patch, (patch_size, patch_size),
                                    interpolation=cv2.INTER_NEAREST)
        mask_full[i:i+patch_size, j:j+patch_size] = pred_patch * 255
   
    if pad_h > 0 or pad_w > 0:
        mask_full = mask_full[:h, :w]
   
    return mask_full
def predict_mask_for_new_image(model, img_path, preprocess_method='unprocessed'):
    img = imread(img_path)
    if len(img.shape) == 2:  
        img = np.stack([img]*3, axis=-1)
       
    original_h, original_w = img.shape[:2]
    mask = predict_full_mask(model, img, preprocess_method)
    if mask.shape[0] != original_h or mask.shape[1] != original_w:
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
   
    return mask
for method in preprocess_methods:
    pred_masks = []
    for fold in range(num_folds):
        model_path = f'best_model_{method}_fold{fold}.keras'
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'combined_loss': combined_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef}
        )
        mask = predict_mask_for_new_image(model, "imgs/img1.jpg", preprocess_method=method)
        pred_masks.append(mask.astype(np.float32) / 255.0)
   
    if pred_masks:
        avg_mask = np.mean(pred_masks, axis=0)
        avg_mask = (avg_mask > 0.5).astype(np.uint8) * 255
        imsave(f"unetpp_mask_resnet_{method}.png", avg_mask)
print("Predictions saved for all preprocessing methods:")
for method in preprocess_methods:
    print(f" - unetpp_mask_resnet_{method}.png")
def show_preprocessing_effects(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.figure(figsize=(16, 8))
   
    unprocessed = apply_preprocessing(img, 'unprocessed')
    plt.subplot(2, 5, 1)
    plt.imshow(unprocessed, cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.title('Unprocessed')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    h, w = unprocessed.shape
    plt.xticks(np.linspace(0, w, 6, dtype=int))
    plt.yticks(np.linspace(0, h, 6, dtype=int))
    plt.ylim(h, 0)
   
    plt.subplot(2, 5, 6)
    n, bins, patches = plt.hist(unprocessed.ravel(), bins=256, color='blue', range=(0, 255))
    max_n = np.max(n)
    num_yticks = 5
    yticks = np.linspace(0, max_n, num_yticks, dtype=int)
    plt.yticks(yticks)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Hist Unprocessed')
    plt.ylim(0, max_n)
    plt.xlim(0, 255)
    plt.xticks(np.linspace(0, 255, 6, dtype=int))
   
    eq = apply_preprocessing(img, 'equalization')
    plt.subplot(2, 5, 2)
    plt.imshow(eq, cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.title('Histogram Equalized')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    h, w = eq.shape
    plt.xticks(np.linspace(0, w, 6, dtype=int))
    plt.yticks(np.linspace(0, h, 6, dtype=int))
    plt.ylim(h, 0)
   
    plt.subplot(2, 5, 7)
    n, bins, patches = plt.hist(eq.ravel(), bins=256, color='green', range=(0, 255))
    max_n = np.max(n)
    num_yticks = 5
    yticks = np.linspace(0, max_n, num_yticks, dtype=int)
    plt.yticks(yticks)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Hist Equalized')
    plt.ylim(0, max_n)
    plt.xlim(0, 255)
    plt.xticks(np.linspace(0, 255, 6, dtype=int))
   
    stretch = apply_preprocessing(img, 'stretching')
    plt.subplot(2, 5, 3)
    plt.imshow(stretch, cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.title('Contrast Stretching')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    h, w = stretch.shape
    plt.xticks(np.linspace(0, w, 6, dtype=int))
    plt.yticks(np.linspace(0, h, 6, dtype=int))
    plt.ylim(h, 0)
   
    plt.subplot(2, 5, 8)
    n, bins, patches = plt.hist(stretch.ravel(), bins=256, color='orange', range=(0, 255))
    max_n = np.max(n)
    num_yticks = 5
    yticks = np.linspace(0, max_n, num_yticks, dtype=int)
    plt.yticks(yticks)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Hist Stretching')
    plt.ylim(0, max_n)
    plt.xlim(0, 255)
    plt.xticks(np.linspace(0, 255, 6, dtype=int))
   
    clahe = apply_preprocessing(img, 'clahe')
    plt.subplot(2, 5, 4)
    plt.imshow(clahe, cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.title('CLAHE')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    h, w = clahe.shape
    plt.xticks(np.linspace(0, w, 6, dtype=int))
    plt.yticks(np.linspace(0, h, 6, dtype=int))
    plt.ylim(h, 0)
   
    plt.subplot(2, 5, 9)
    n, bins, patches = plt.hist(clahe.ravel(), bins=256, color='red', range=(0, 255))
    max_n = np.max(n)
    num_yticks = 5
    yticks = np.linspace(0, max_n, num_yticks, dtype=int)
    plt.yticks(yticks)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Hist CLAHE')
    plt.ylim(0, max_n)
    plt.xlim(0, 255)
    plt.xticks(np.linspace(0, 255, 6, dtype=int))
   
    low_contrast = apply_preprocessing(img, 'low_contrast')
    plt.subplot(2, 5, 5)
    plt.imshow(low_contrast, cmap='gray', origin='upper')
    plt.gca().invert_yaxis()
    plt.title('Low Contrast')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    h, w = low_contrast.shape
    plt.xticks(np.linspace(0, w, 6, dtype=int))
    plt.yticks(np.linspace(0, h, 6, dtype=int))
    plt.ylim(h, 0)
   
    plt.subplot(2, 5, 10)
    n, bins, patches = plt.hist(low_contrast.ravel(), bins=256, color='purple', range=(0, 255))
    max_n = np.max(n)
    num_yticks = 5
    yticks = np.linspace(0, max_n, num_yticks, dtype=int)
    plt.yticks(yticks)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Hist Low Contrast')
    plt.ylim(0, max_n)
    plt.xlim(0, 255)
    plt.xticks(np.linspace(0, 255, 6, dtype=int))
   
    plt.tight_layout()
    plt.savefig('unetpp_resnet_preprocessing_effects.png')
    plt.close()
    print("Preprocessing effects plot saved to unetpp_resnet_preprocessing_effects.png")
   
   
   
if all_images:
    show_preprocessing_effects(all_images[0])
for method in preprocess_methods:
    print(f"\nGenerating predictions and pixel hist for method: {method}")
    model_to_use = None
    try:
        model_to_use = tf.keras.models.load_model(
            best_fold_model[method],
            custom_objects={'combined_loss': combined_loss, 'dice_coef': dice_coef, 'iou_coef': iou_coef}
        )
        print(f"Loaded best model {best_fold_model[method]}")
    except Exception as e:
        print(f"Could not load best model for {method}: {e}")
        continue
    sample_img_path = "imgs/img1.jpg"
    if os.path.exists(sample_img_path):
        try:
            pred_mask = predict_mask_for_new_image(model_to_use, sample_img_path, preprocess_method=method)
            out_pred_name = f"unetpp_resnet_prediction_{method}.png"
            imsave(out_pred_name, pred_mask)
            print(f"Saved prediction for sample image -> {out_pred_name}")
        except Exception as e:
            print(f"Error predicting sample image for {method}: {e}")
    if os.path.exists(sample_img_path):
        try:
            img_raw = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
            if img_raw is None:
                img_raw = imread(sample_img_path)
                if len(img_raw.shape) == 3:
                    img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            img_pre = apply_preprocessing(img_raw.copy(), method)
            img_pre_u8 = np.clip(img_pre, 0, 255).astype(np.uint8)
            hist_counts, bin_edges = np.histogram(img_pre_u8.ravel(), bins=256, range=(0,255))
            df_hist = pd.DataFrame({
                'intensity': np.arange(256),
                'pixel_count': hist_counts
            })
            hist_csv_name = f'unetpp_resnet_pixel_hist_{method}.csv'
            df_hist.to_csv(hist_csv_name, index=False)
            print(f"Saved pixel intensity histogram CSV -> {hist_csv_name}")
        except Exception as e:
            print(f"Error computing/saving pixel histogram for {method}: {e}")
    else:
        print(f"Sample image {sample_img_path} not found, skipping pixel histogram for {method}.")
