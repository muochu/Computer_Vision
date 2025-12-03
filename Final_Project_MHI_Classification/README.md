# MHI Activity Classification

Motion History Image (MHI) based activity classification system for recognizing human actions in video.

## Requirements

- Python 3.7+
- OpenCV (`cv2`)
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

Install dependencies:
```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
```

## Project Structure

```
Final_Project_MHI_Classification/
├── data/                    # Dataset (AVI files organized by action)
├── src/                     # Source code
├── results/                 # Output directory
├── outputs/                 # Generated MHIs and videos
└── README.md
```

## Usage

### Training

Train the classifier on the KTH Actions dataset:

```bash
python src/main.py --data_dir data --seq_file 00sequence.txt --tau 15 --theta 30 --method knn --use_scaled
```

### Prediction

Predict action on an arbitrary video:

```bash
python src/predict.py path/to/video.avi --classifier results/classifier.pkl
```

## Implementation Details

### MHI Generation
- Frame differencing with threshold θ
- Morphological opening for noise removal
- MHI construction with decay parameter τ

### Feature Extraction
- Regular moments: M_ij = Σ Σ x^i y^j I(x,y)
- Central moments: μ_pq = Σ Σ (x-x̄)^p (y-ȳ)^q I(x,y)
- Scale-invariant moments: ν_pq = μ_pq / μ_00^(1+(p+q)/2)
- Features from moments: {20, 11, 02, 30, 21, 12, 03, 22}

### Classification
- k-NN or SVM classifier from Scikit-Learn
- Train/validation/test split: 8/8/9 subjects

## Notes

- All paths are relative to the project directory
- Code is platform-independent (works on Linux/Windows/Mac)
- Hu moments implemented from scratch (no `cv2.HuMoments`)
- MHI generation implemented from scratch (no pre-built functions)

