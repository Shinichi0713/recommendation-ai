import os
import shutil
from glob import glob

!git clone https://github.com/DHW-Master/NEU_Seg.git

# 1. 設定：新しいディレクトリ構造の定義
base_dir = "./datasets/NEU_Seg_Custom"
structure = [
    "train/images", "train/masks",
    "test/images", "test/masks"
]

for path in structure:
    os.makedirs(os.path.join(base_dir, path), exist_ok=True)

# 2. ファイルの移動（またはコピー）
# 元のディレクトリ: /content/NEU_Seg/images と /content/NEU_Seg/annotations
src_img_dir = "/content/NEU_Seg/images"
src_ann_dir = "/content/NEU_Seg/annotations"

# NEU_Segは train/test の分割が明示的にフォルダ分けされていない場合があるため、
# ここでは「全データを一度 train に入れ、一部を test に移動」する汎用的な処理を行います。

def organize_neu_data():
    dirs_target = ["test", "training"]
    for dir_target in dirs_target:
        all_images = sorted(glob(os.path.join(src_img_dir, dir_target, "*.jpg")))
        
        print(f"Total images found: {len(all_images)}")
        
        # 8:2 で Train と Test に分割する例
        split_idx = int(len(all_images) * 0.8)
        train_imgs = all_images[:split_idx]
        test_imgs = all_images[split_idx:]
        
        def move_files(file_list, target_sub_dir):
            for img_path in file_list:
                fname = os.path.basename(img_path)
                # 対応するマスクファイル名を探す (拡張子は .png の場合が多い)
                mask_name = fname.replace(".jpg", ".png")
                
                # 画像のコピー
                shutil.copy(img_path, os.path.join(base_dir, target_sub_dir, "images", fname))
                
                # マスクのコピー (存在する場合のみ)
                # annotations 内のフォルダ構成に合わせて検索
                potential_mask_paths = [
                    os.path.join(src_ann_dir, mask_name),
                    os.path.join(src_ann_dir, "test", mask_name),
                    os.path.join(src_ann_dir, "train", mask_name) # リポジトリ構造に合わせる
                ]
                
                for m_path in potential_mask_paths:
                    if os.path.exists(m_path):
                        shutil.copy(m_path, os.path.join(base_dir, target_sub_dir, "masks", mask_name))
                        break

        print("Copying to Train...")
        move_files(train_imgs, "train")
        print("Copying to Test...")
        move_files(test_imgs, "test")
        print("Data organization complete!")

organize_neu_data()


class NEUSegDataset(Data.Dataset):
    def __init__(self, base_dir, mode='train', base_size=256):
        self.img_dir = os.path.join(base_dir, mode, 'images')
        self.mask_dir = os.path.join(base_dir, mode, 'masks')
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg')])
        self.base_size = base_size
        self.transform = transforms.Compose([
            transforms.Resize((base_size, base_size)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        mask_path = os.path.join(self.mask_dir, name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L') # マスクもグレースケールで読み込み

        image = self.transform(image)
        mask = self.transform(mask)
        
        # マスクを0or1のバイナリにする（SoftIoU用）
        mask = (mask > 0.5).float()

        return image, mask

    def __len__(self):
        return len(self.img_names)