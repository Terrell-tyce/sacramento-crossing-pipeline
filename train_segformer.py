import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
from datasets import Dataset as HFDataset
from PIL import Image
import evaluate
from sklearn.preprocessing import LabelEncoder
import albumentations as A

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_PATH = r"C:\Users\tenoru\Downloads"  # Change this to your dataset folder
IMAGES_DIR = os.path.join(DATASET_PATH, "high entropy images")
MASKS_DIR = os.path.join(DATASET_PATH, "undriveable masks")
MASKS_DIR= os.path.join(MASKS_DIR, "SegmentationClass")  # Folder with segmentation masks
OUTPUT_DIR = "./segformer_output"

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"  # b0 is small/fast. Use b1-b5 for larger models
NUM_CLASSES = 2  # background + undriveable 
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
BATCH_SIZE = 8
SEED = 42

# ============================================================================
# STEP 1: CREATE CUSTOM DATASET CLASS
# ============================================================================

class SegmentationDataset(Dataset):
    """
    Custom dataset that loads images and their corresponding masks.
    Pairs images from JPEGImages/ with masks from SegmentationClass/
    Applies augmentations to training set only.
    """
    
    def __init__(self, images_dir, masks_dir, processor, train=True, train_split=0.8):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.processor = processor
        
        # Get all image filenames
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Split into train/val
        num_train = int(len(self.image_files) * train_split)
        if train:
            self.image_files = self.image_files[:num_train]
        else:
            self.image_files = self.image_files[num_train:]
        
        # Augmentation pipeline (only applied to training set)
        self.train = train
        if train:
            self.augment = A.Compose([
                # Geometric augmentations (applied to both image and mask)
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.1),
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
                
                # Color augmentations (applied to image only)
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussNoise(p=0.1),
                A.RandomFog(p=0.1),
            ], is_check_shapes=False)
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(image_path).convert("RGB")
        
        # Load corresponding mask (same filename but from masks directory)
        mask_filename = os.path.splitext(img_filename)[0] + ".png"
        mask_path = os.path.join(self.masks_dir, mask_filename)
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale (single channel)
        
        # Convert to numpy arrays for augmentation
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Apply augmentations (training set only)
        if self.augment is not None:
            augmented = self.augment(image=image_np, mask=mask_np)
            image_np = augmented['image']
            mask_np = augmented['mask']
        
        # Convert back to PIL for processor
        image = Image.fromarray(image_np.astype('uint8'))
        mask = Image.fromarray(mask_np.astype('uint8'))
        
        # Process image and mask with the processor
        # The processor resizes, normalizes, and prepares the image for the model
        processed = self.processor(images=image, segmentation_maps=mask, 
                                    return_tensors="pt")
        
        # Remove batch dimension that processor adds
        for key, val in processed.items():
            processed[key] = val.squeeze()
        
        return processed

# ============================================================================
# STEP 2: LOAD DATASET AND CREATE DATA LOADERS
# ============================================================================

# Initialize processor (handles image resizing, normalization, etc.)
processor = SegformerImageProcessor.from_pretrained(MODEL_ID)

# Create train and validation datasets
train_dataset = SegmentationDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    processor=processor,
    train=True,
    train_split=0.8
)

val_dataset = SegmentationDataset(
    images_dir=IMAGES_DIR,
    masks_dir=MASKS_DIR,
    processor=processor,
    train=False,
    train_split=0.8
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ============================================================================
# STEP 3: DEFINE METRIC FOR EVALUATION
# ============================================================================

def compute_metrics(eval_pred):
    """
    Compute Mean Intersection over Union (mIoU) - standard segmentation metric.
    mIoU = mean of (intersection / union) for each class
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Flatten spatial dimensions
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    
    # Calculate IoU for each class
    metric = evaluate.load("mean_iou")
    results = metric.compute(
        predictions=predictions,
        references=labels,
        num_labels=NUM_CLASSES,
        ignore_index=255  # Ignore unlabeled pixels
    )
    
    return results

# ============================================================================
# STEP 4: LOAD PRE-TRAINED MODEL
# ============================================================================

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True  # Allows changing number of output classes
)

print(f"Model loaded: {MODEL_ID}")
print(f"Model parameters: {model.num_parameters():,}")

# ============================================================================
# STEP 5: DEFINE TRAINING ARGUMENTS
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=3,  # Only keep last 3 checkpoints
    eval_strategy="epoch",  # Evaluate at end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    push_to_hub=False,
    seed=SEED,
    remove_unused_columns=False,
    dataloader_pin_memory=True,  # Speed up data loading with GPU
)

# ============================================================================
# STEP 6: CREATE TRAINER AND TRAIN
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=None,  # Use default collator
)

print("Starting training...")
trainer.train()

# ============================================================================
# STEP 7: SAVE AND EVALUATE
# ============================================================================

print("Training complete!")
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

# Evaluate on validation set
print("\nEvaluating on validation set...")
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# ============================================================================
# OPTIONAL: INFERENCE ON A SINGLE IMAGE
# ============================================================================

def predict_segmentation(image_path, model, processor):
    """
    Load an image and run inference to get segmentation prediction.
    """
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get class prediction for each pixel
    logits = outputs.logits  # Shape: (batch, num_classes, height, width)
    predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
    
    return predicted_mask

# Example usage:
# mask = predict_segmentation("path/to/test_image.jpg", model, processor)
# print("Predicted mask shape:", mask.shape)
# print("Unique classes in prediction:", np.unique(mask))