
"""
IMPROVED Data Preprocessing for ASL Dataset - OPTIMIZED
"""
import os
import shutil
from pathlib import Path
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ✅ استيراد دوال المعالجة المسبقة للمودلات
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from utils import ensure_dir, save_class_indices, get_classes_from_directory

# ✅ استورد كل حاجة من config
from config import (
    RAW_DATA_DIR, DATA_DIR, IMG_SIZE, BATCH_SIZE, SEED,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE, ZOOM_RANGE, BRIGHTNESS_RANGE, CHANNEL_SHIFT_RANGE
)

# Set random seeds
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def split_dataset():
    """Split dataset into train/val/test sets"""
    print("\n" + "=" * 60)
    print("✂️  Splitting Dataset into Train/Val/Test")
    print("=" * 60)

    # Create directories
    for split in ['train', 'val', 'test']:
        ensure_dir(DATA_DIR / split)

    classes = get_classes_from_directory(RAW_DATA_DIR)

    split_stats = []
    all_train_labels = []

    for cls in classes:
        cls_path = RAW_DATA_DIR / cls
        images = list(cls_path.glob('*.jpg')) + list(cls_path.glob('*.png')) + list(cls_path.glob('*.jpeg'))

        # Split data
        train_imgs, temp_imgs = train_test_split(
            images, train_size=TRAIN_RATIO, random_state=SEED
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=VAL_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=SEED
        )

        # Copy images
        for split, img_list in [('train', train_imgs),
                                ('val', val_imgs),
                                ('test', test_imgs)]:
            split_cls_path = DATA_DIR / split / cls
            ensure_dir(split_cls_path)

            for img_path in img_list:
                shutil.copy(img_path, split_cls_path / img_path.name)

        # Store labels for class weight calculation
        all_train_labels.extend([classes.index(cls)] * len(train_imgs))

        # ✅✅ التصحيح الأول: ملء قائمة الإحصائيات
        split_stats.append({
            'Class': cls,
            'Train': len(train_imgs),
            'Val': len(val_imgs),
            'Test': len(test_imgs),
            'Total': len(images)
        })

        print(f"✅ {cls:12s} | Train: {len(train_imgs):4d} | "
              f"Val: {len(val_imgs):4d} | Test: {len(test_imgs):4d}")

    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(all_train_labels),
        y=all_train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))

    # Save class weights
    weights_path = DATA_DIR / 'class_weights.json'
    with open(weights_path, 'w') as f:
        json.dump(class_weights_dict, f, indent=4)

    # Summary
    # ✅ التصحيح الثاني: حساب الإجمالي بناءً على القائمة الممتلئة الآن
    total_train = sum(s['Train'] for s in split_stats)
    total_val = sum(s['Val'] for s in split_stats)
    total_test = sum(s['Test'] for s in split_stats)
    total_all = sum(s['Total'] for s in split_stats)

    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   Train: {total_train} ({total_train / total_all * 100:.1f}%)")
    print(f"   Val:   {total_val} ({total_val / total_all * 100:.1f}%)")
    print(f"   Test:  {total_test} ({total_test / total_all * 100:.1f}%)")
    print(f"   Total: {total_all}")
    print(f"✅ Class weights saved to: {weights_path}")
    print("=" * 60 + "\n")

    return class_weights_dict


def get_preprocessing_settings(model_name):
    """Helper function to get model-specific preprocessing function and rescale value."""
    preprocessing_function = None
    rescale_value = 1. / 255 # القيمة الافتراضية
    
    if model_name == 'EfficientNetB0':
        preprocessing_function = efficientnet_preprocess
        rescale_value = None
    elif model_name == 'ResNet50':
        preprocessing_function = resnet_preprocess
        rescale_value = None
    elif model_name == 'InceptionV3':
        preprocessing_function = inception_preprocess
        rescale_value = None
    
    return preprocessing_function, rescale_value

def create_data_generators(model_name):
    """Create OPTIMIZED data generators with correct model-specific preprocessing"""
    print("\n" + "=" * 60)
    print(f"🎨 Creating OPTIMIZED Data Generators for {model_name}")
    print("=" * 60)
    
    # ✅ 1. الحصول على إعدادات المعالجة الصحيحة
    preprocessing_function, rescale_value = get_preprocessing_settings(model_name)
    
    if rescale_value is None:
        print(f"✅ Using model-specific preprocessing function for {model_name} (No external rescale)")
    else:
        print("⚠️ Warning: Using default rescaling (1/255) as a model-specific function was not found.")

    # ✅ 2. إعداد المولدات مع القيم الصحيحة
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=rescale_value,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=True,
        brightness_range=BRIGHTNESS_RANGE,
        fill_mode='nearest',
        channel_shift_range=CHANNEL_SHIFT_RANGE,
    )

    # Validation and test data (only necessary preprocessing)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=rescale_value
    )

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR / 'train',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        DATA_DIR / 'val',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        DATA_DIR / 'test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\n✅ Training samples:   {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples:       {test_generator.samples}")
    print(f"✅ Number of classes:  {train_generator.num_classes}")
    print(f"✅ Image size:         {IMG_SIZE}x{IMG_SIZE}")
    print(f"✅ Batch size:         {BATCH_SIZE}")

    # Save class indices
    class_indices_path = DATA_DIR / 'class_indices.json'
    save_class_indices(train_generator.class_indices, class_indices_path)

    print("=" * 60 + "\n")

    return train_generator, val_generator, test_generator


def load_class_weights():
    """Load precomputed class weights"""
    weights_path = DATA_DIR / 'class_weights.json'
    if weights_path.exists():
        with open(weights_path, 'r') as f:
            return json.load(f)
    return None


def main():
    """Run preprocessing pipeline"""
    print("\n" + "🚀" * 30)
    print("Starting OPTIMIZED Data Preprocessing Pipeline")
    print("🚀" * 30 + "\n")

    # Step 1: Split dataset and get class weights
    class_weights = split_dataset()

    # Step 2: Test creation of generators (Demonstration)
    try:
        print("\n🧪 Testing generator creation for ResNet50...")
        # يتم إنشاء المولدات هنا فقط للاختبار - لن يتم استخدامها في التدريب الفعلي
        train_gen, val_gen, test_gen = create_data_generators('ResNet50')
        print("✅ Generator test successful.")
    except Exception as e:
        print(f"❌ Could not test generators (ensure utils is correct): {e}")


    print(f"\n📊 Class weights: {class_weights}")

    print("\n" + "✅" * 30)
    print("OPTIMIZED Data Preprocessing Complete!")
    print("✅" * 30 + "\n")


if __name__ == "__main__":
    main()
