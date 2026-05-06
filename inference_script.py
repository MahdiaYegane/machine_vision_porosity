import os
import numpy0 as np
import tensorflow as tf
from skimage.io import imread, imsave
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import zipfile
import tempfile
import shutil

os.environ['TF_USE_LEGACY_KERAS'] = '1'

SEED = 42
PATCH_SIZE = 128
BATCH_SIZE = 8
num_folds = 7

np.random.seed(SEED)
tf.random.set_seed(SEED)


def calculate_porosity(mask):
    binary = (mask > 127).astype(np.uint8) * 255
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.size
    return (white_pixels / total_pixels) * 100


def decoder_block(input_tensor, skip_tensor, filters, last_block=False):
    x = Conv2DTranspose(filters, (3, 3), strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    if not last_block:
        x = Concatenate()([x, skip_tensor])
    
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def build_resnet50_unet(input_shape=(128, 128, 3)):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    
    skip_connection_names = [
        "conv1_relu",
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out",
    ]
    skip_connections = [base_model.get_layer(name).output for name in skip_connection_names]
    encoder_output = base_model.output
    x = encoder_output
    x = decoder_block(x, skip_connections[3], 512)
    x = decoder_block(x, skip_connections[2], 256)
    x = decoder_block(x, skip_connections[1], 128)
    x = decoder_block(x, skip_connections[0], 64)
    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model


def load_model_trained_weights(model_path):
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        weights_h5 = os.path.join(temp_dir, 'model.weights.h5')
        if not os.path.exists(weights_h5):
            raise FileNotFoundError("Weights.h5 not found in .keras")
        
        model = build_resnet50_unet()
        model.load_weights(weights_h5)
        print(f"Loaded full trained weights from {model_path}")
        return model
    except Exception as e:
        print(f"Load error: {e}")
        raise
    finally:
        shutil.rmtree(temp_dir)


def unfreeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
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


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = 1 - dice_coef(y_true, y_pred)
    return bce + dice


def apply_preprocessing(img, method='unprocessed'):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'unprocessed':
        return img
    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        return img
    else:
        raise ValueError(f"Unknown method: {method}")


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
    
    img = apply_preprocessing(img, preprocess_method)
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
                    patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1]), (0, 0)), mode='reflect')
                else:
                    patch = np.pad(patch, ((0, patch_size - patch.shape[0]), (0, patch_size - patch.shape[1])), mode='reflect')
            patches.append(patch)
            positions.append((i, j))
    
    patches = np.array(patches)
    preds = model.predict(patches, batch_size=BATCH_SIZE, verbose=0)
    preds = (preds > 0.5).astype(np.uint8)
    
    mask_full = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for idx, (i, j) in enumerate(positions):
        pred_patch = preds[idx].squeeze()
        if pred_patch.shape[0] != patch_size or pred_patch.shape[1] != patch_size:
            pred_patch = cv2.resize(pred_patch, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
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


def run_inference(input_image_path="imgs/img1.jpg", output_dir="inference_results", reference_mask_path=None):
    os.makedirs(output_dir, exist_ok=True)
    methods = ['clahe']
    
    input_img = imread(input_image_path)
    if len(input_img.shape) == 3:
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    original_h, original_w = input_img.shape
    
    ref_mask = None
    gt_por = 0
    if reference_mask_path and os.path.exists(reference_mask_path):
        ref = imread(reference_mask_path)
        if len(ref.shape) == 3:
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref_mask = ref
        gt_por = calculate_porosity(ref_mask)
        print(f"Ground Truth Porosity: {gt_por:.2f}%")
    else:
        print("No reference mask provided. Skipping GT porosity calculation.")
    
    for method in methods:
        pred_masks = []
        
        print(f"Generating predictions for {method}...")
        
        for fold in range(num_folds):
            model_path = f'best_model_{method}_fold{fold}.keras'
            if not os.path.exists(model_path):
                print(f"Skipping {model_path} (not found)")
                continue
            
            try:
                model = load_model_trained_weights(model_path)
                model = unfreeze_model(model)
                
                dummy = np.random.rand(1, 128, 128, 3).astype(np.float32)
                dummy = preprocess_input(dummy)
                pred = model.predict(dummy, verbose=0)
                print(f"Fold {fold}: Sanity pred range [{pred.min():.3f}, {pred.max():.3f}]")
                
                mask = predict_mask_for_new_image(model, input_image_path, method)
                pred_masks.append(mask.astype(np.float32) / 255.0)
                print(f"Fold {fold}: Mask non-zero ratio {np.sum(mask > 0) / mask.size:.4f}")
                
            except Exception as e:
                print(f"Error fold {fold}: {e}")
                continue
        
        if pred_masks:
            avg_mask = np.mean(pred_masks, axis=0)
            avg_mask = (avg_mask > 0.5).astype(np.uint8) * 255
            output_path = os.path.join(output_dir, f"unet_mask_resnet50_{method}.png")
            imsave(output_path, avg_mask)
            
            pred_por = calculate_porosity(avg_mask)
            diff_por = abs(pred_por - gt_por)
            print(f"Saved {output_path} | Predicted Porosity: {pred_por:.2f}% | Difference from GT: {diff_por:.2f}%")
            
            if ref_mask is not None:
                plt.figure(figsize=(18, 8))
                ax1 = plt.subplot(1, 3, 1)
                plt.imshow(input_img, cmap='gray')
                plt.title('Input Image')
                plt.axis('off')
                
                ax2 = plt.subplot(1, 3, 2)
                plt.imshow(avg_mask, cmap='gray')
                plt.title(f'Predicted Mask (ResNet50 + {method.upper()})')
                plt.axis('off')
                
                ax3 = plt.subplot(1, 3, 3)
                h, w = avg_mask.shape
                diff = np.zeros((h, w, 3), dtype=np.uint8)
                intersection = np.logical_and(ref_mask > 127, avg_mask > 127)
                only_ref = np.logical_and(ref_mask > 127, avg_mask <= 127)
                only_pred = np.logical_and(avg_mask > 127, ref_mask <= 127)
                diff[intersection] = [255, 0, 0]
                diff[only_ref] = [0, 0, 255]
                diff[only_pred] = [0, 255, 0]
                
                plt.imshow(diff)
                plt.title('Mask Difference Visualization\n(Red: Overlap, Blue: Only GT, Green: Only Pred)')
                plt.axis('off')
                
                text_str = f'Ground Truth Porosity: {gt_por:.2f}%\nResNet50 {method.upper()} Porosity: {pred_por:.2f}%\nPorosity difference: {diff_por:.2f}%'
                plt.text(0.5, -0.1, text_str,
                         transform=ax3.transAxes, fontsize=12, ha='center', va='top',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))
                
                legend_elements = [
                    Patch(facecolor='red', label='overlap'),
                    Patch(facecolor='blue', label='only ground_truth mask'),
                    Patch(facecolor='green', label='only prediction mask')
                ]
                ax3.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1), ncol=1)
                
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.2, top=0.9, hspace=0.3, wspace=0.05)
            else:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(input_img, cmap='gray')
                plt.title('Input')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(avg_mask, cmap='gray')
                plt.title('Mask')
                plt.axis('off')
            
            vis_path = os.path.join(output_dir, f"vis_{method}.png")
            plt.savefig(vis_path)
            plt.close()
            print(f"Vis: {vis_path}")
        else:
            print("No successful loads.")


if __name__ == "__main__":
    INPUT_IMAGE_PATH = "imgs/img5.jpg"
    REFERENCE_MASK_PATH = "cleaned_masks/mask5.jpg"
    if os.path.exists(INPUT_IMAGE_PATH):
        run_inference(INPUT_IMAGE_PATH, reference_mask_path=REFERENCE_MASK_PATH)
    else:
        print(f"Update {INPUT_IMAGE_PATH}")
