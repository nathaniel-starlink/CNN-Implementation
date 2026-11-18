import os
import glob
import random
import cv2

# Konfigurasi Path (Relatif dari folder algoritma_cnn/)
DATASET_ROOT = "../dataset"
OUT_ROOT = "../dataset/processed_64x64"
IMG_SIZE = 64
TRAIN_RATIO = 0.8  # 80% train, 20% valid
SEED = 42

# Mapping nama sub-folder sumber ke nama folder kelas target
SOURCES = {
    "intact_cans": "0_sempurna",
    "damaged_cans": "1_penyok"
}

def extract_crops_from_source(source_folder, target_class_name):
    source_path = os.path.join(DATASET_ROOT, source_folder)
    if not os.path.exists(source_path):
        print(f"[PERINGATAN] Folder tidak ditemukan: {source_path}")
        return []

    # Cari seluruh gambar di sub-folder manapun (train/valid/test)
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(glob.glob(os.path.join(source_path, "**", "images", ext), recursive=True))

    cropped_items = []
    for img_path in img_files:
        # Cari file label .txt YOLOv8 yang bersesuaian
        label_path = img_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
        label_path = os.path.splitext(label_path)[0] + ".txt"

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h_img, w_img, _ = img.shape
        # Konversi ke 1-channel Grayscale 8-bit (sesuai Input Buffer BRAM FPGA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        with open(label_path, "r") as f:
            lines = f.readlines()

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            x_center, y_center, w_box, h_box = map(float, parts[1:5])

            # Konversi koordinat normalisasi YOLO ke piksel absolut
            x1 = max(0, int((x_center - w_box / 2.0) * w_img))
            y1 = max(0, int((y_center - h_box / 2.0) * h_img))
            x2 = min(w_img, int((x_center + w_box / 2.0) * w_img))
            y2 = min(h_img, int((y_center + h_box / 2.0) * h_img))

            crop = gray[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            # Resize ke 64x64 piksel (4.096 byte per gambar)
            resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            filename = f"{source_folder}_{base_name}_obj{idx}.png"
            cropped_items.append((filename, resized, target_class_name))

    return cropped_items

def main():
    random.seed(SEED)
    total_train, total_val = 0, 0

    for src_folder, class_name in SOURCES.items():
        items = extract_crops_from_source(src_folder, class_name)
        random.shuffle(items)

        split_idx = int(len(items) * TRAIN_RATIO)
        splits = {
            "train": items[:split_idx],
            "valid": items[split_idx:]
        }

        for split_name, split_items in splits.items():
            out_dir = os.path.join(OUT_ROOT, split_name, class_name)
            os.makedirs(out_dir, exist_ok=True)

            for fname, img_data, _ in split_items:
                cv2.imwrite(os.path.join(out_dir, fname), img_data)

        print(f"Kelas [{class_name}] dari '{src_folder}': Total={len(items)} | Train={len(splits['train'])} | Valid={len(splits['valid'])}")
        total_train += len(splits["train"])
        total_val += len(splits["valid"])

    print(f"\nSelesai! Total dataset siap latih -> Train: {total_train}, Valid: {total_val}")
    print(f"Lokasi output: {os.path.abspath(OUT_ROOT)}")

if __name__ == "__main__":
    main()