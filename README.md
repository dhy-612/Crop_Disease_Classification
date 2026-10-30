# Crop Disease Classification using Deep Learning

## 1. Project Overview

This project builds a 15-class crop leaf disease classification system
using the PlantVillage image dataset and transfer learning with
MobileNetV2.

The project covers exploratory data analysis (EDA), image preprocessing
and augmentation, model training, evaluation, error analysis, and a
Streamlit application for image-based predictions.

**Objective:** Given an image of a crop leaf, predict its corresponding
disease or healthy class and display the model's top predictions with
confidence scores.

## 2. Dataset

-   **Dataset:** PlantVillage
-   **Total images:** 20,638
-   **Number of classes:** 15
-   **Train split:** 14,437 images (approximately 70%)
-   **Validation split:** 3,100 images (approximately 15%)
-   **Test split:** 3,101 images (approximately 15%)

The dataset contains pepper, potato, and tomato leaf images. Healthy
categories are retained as separate classes.

> The dataset is not included in this repository. Download the
> PlantVillage dataset separately and update the dataset path in the
> notebook before running the training workflow.

## 3. Project Structure

``` text
Crop_Disease_Classification/
│
├── notebooks/
│   └── crop_disease_classification.ipynb
│
├── model_artifacts/
│   └── best_mobilenetv2.weights.h5
│
├── app.py
├── class_names.json
├── requirements.txt
└── README.md
```

Adjust the notebook filename or folder names above if your local project
uses a different structure.

## 4. Model and Methodology

  -----------------------------------------------------------------------
  Component                           Approach
  ----------------------------------- -----------------------------------
  Model                               MobileNetV2

  Learning approach                   Transfer learning

  Input image size                    224 × 224 × 3

  Output layer                        Dense layer with 15 units and
                                      softmax activation

  Loss function                       Sparse categorical cross-entropy

  Optimizer                           Adam

  Data augmentation                   Horizontal flip, rotation, and zoom

  Evaluation                          Accuracy, precision, recall,
                                      F1-score, and confusion matrix
  -----------------------------------------------------------------------

The MobileNetV2 feature extractor was initialized with ImageNet weights
and frozen during the initial training stage. A custom classification
head was trained for the 15 target classes.

## 5. Model Performance

The final saved checkpoint was evaluated on the held-out test set.

  Metric                 Test result
  -------------------- -------------
  Test accuracy           **89.78%**
  Macro precision             90.13%
  Macro recall                88.13%
  Macro F1-score              88.50%
  Weighted precision          90.16%
  Weighted recall             89.78%
  Weighted F1-score           89.61%
  Test images                  3,101

### Key evaluation observations

-   The model performs well on several classes, including Pepper
    Bacterial Spot, Potato Early Blight, Potato Healthy, and Tomato
    Yellow Leaf Curl Virus.
-   Lower recall was observed for Tomato Early Blight (**57.33%**) and
    Tomato Mosaic Virus (**66.07%**).
-   The confusion matrix and sample-level prediction analysis show that
    visually similar leaf symptoms can lead to misclassification.

## 6. Business Recommendation

MobileNetV2 is a practical candidate for a mobile-oriented crop disease
screening application because it offers a relatively lightweight
architecture and achieved 89.78% accuracy on the held-out test set. The
Streamlit prototype demonstrates how a user can upload a leaf image and
receive the predicted class, confidence score, and top-three
predictions. However, the model should be treated as a decision-support
tool rather than an autonomous diagnostic system at this stage. In
particular, the lower recall for Tomato Early Blight and Tomato Mosaic
Virus means some affected leaves may be missed. Before field deployment,
the model should be validated on images captured under real farm
conditions, improved using more representative data for weaker classes,
and tested after mobile optimization such as TensorFlow Lite conversion.
Low-confidence predictions should be escalated for expert review.

## 7. Setup Instructions

### Prerequisites

-   Python 3.10 or a compatible Python version supported by your
    TensorFlow installation
-   pip
-   Git (optional)

### Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, create one containing the
packages used by the notebook and application. For example:

``` text
tensorflow
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
```

For reproducibility, use the TensorFlow/Keras versions with which the
application was tested, and pin the exact working package versions in
`requirements.txt`.

## 8. Running the Notebook

1.  Open the notebook from the `notebooks/` folder in Jupyter Notebook,
    JupyterLab, or VS Code.
2.  Download and extract the PlantVillage dataset.
3.  Update `DATASET_PATH` in the notebook to point to the extracted
    dataset directory.
4.  Run the notebook cells in order to perform EDA, create the
    train/validation/test splits, train the model, and evaluate it.

The notebook contains the code, visualizations, and Markdown
explanations for the complete experimentation workflow.

## 9. Running the Streamlit Application

Ensure that the following files are present:

-   `app.py`
-   `class_names.json`
-   `model_artifacts/best_mobilenetv2.weights.h5`

Run:

``` bash
streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit. Upload a leaf
image and run the prediction to view the predicted class, confidence,
and top-three results.

## 10. Limitations and Future Improvements

-   Evaluate the model on real-world field images with different
    lighting, backgrounds, camera quality, and leaf orientations.
-   Improve performance on classes with lower recall, especially Tomato
    Early Blight and Tomato Mosaic Virus.
-   Explore class-weighted training or targeted data augmentation for
    underperforming classes.
-   Calibrate prediction confidence and define a low-confidence review
    threshold.
-   Convert and benchmark the model using TensorFlow Lite before mobile
    deployment.
-   Monitor model performance after deployment and retrain with reviewed
    field data.

## 11. Deliverables

This project submission includes:

1.  **Presentation:** Project overview, methodology, results, error
    analysis, and business recommendation.
2.  **Jupyter Notebook:** Code, visualizations, and Markdown
    explanations.
3.  **README.md:** Setup instructions, model performance summary, and
    business recommendation.
4.  **Saved model (optional):** Trained MobileNetV2 weights file.
