# README_test_pt

## 1. Install dependencies

```bash
pip install ultralytics opencv-python
```

## 2. Test script

- `sign_test_pt.py`: loads `best.pt` and runs inference on image, image folder, or video.

## 3. Usage

Run in `D:\smartcar-competition\baord\test`:

### 3.1 Single image

```bash
python sign_test_pt.py --source "D:\smartcar-competition\sign\images data\xxx.jpg"
```

### 3.2 Image folder

```bash
python sign_test_pt.py --source "D:\smartcar-competition\sign\images data"
```

### 3.3 Video file

```bash
python sign_test_pt.py --source "D:\smartcar-competition\sign\videos\xxx.mp4"
```

### 3.4 Optional arguments

```bash
python sign_test_pt.py ^
  --weights "D:\smartcar-competition\runs\detect\sign_v2_fixed_100\weights\best.pt" ^
  --source "D:\smartcar-competition\sign\images data" ^
  --imgsz 640 ^
  --conf 0.25 ^
  --save-dir "D:\smartcar-competition\baord\test\results"
```

## 4. Output location

- Default save directory: `D:\smartcar-competition\baord\test\results`
- Single image: `xxx_result.jpg/png/...`
- Image folder: `results\folder_name_result\`
- Video: `xxx_result.mp4`

## 5. How this meets competition test-code requirement

- Provides an independent runnable PT inference script: `sign_test_pt.py`.
- Directly loads submission model `best.pt` for sign recognition testing.
- Supports image, image-folder, and video inputs.
- Saves boxed output results and prints key logs:
  input path, model path, success/failure, and output save path.
- Includes required exception handling:
  missing model file, missing source path, and video open failure.
