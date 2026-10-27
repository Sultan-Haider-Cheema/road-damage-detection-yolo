# Automatic Road Damage Detection from Street Images using Deep Learning

M1 Machine Learning and Data Mining (MLDM) project, Université Jean Monnet Saint-Étienne, 2025–2026.
**Author:** Muhammad Sultan Haidar Cheema

This project compares three YOLO object detectors (**YOLOv5s, YOLOv8n, YOLOv8s**) on the RDD2022 road damage dataset under identical training settings. It also tests **cross-country generalization** by training on Japanese images only and evaluating on six countries. The best model is deployed in a **Streamlit** app.

**Documents:** [Project report (PDF)](docs/Project_Report.pdf) · [Defence slides (PDF)](docs/Defence_Slides.pdf) 
## Main results (RDD2022 test set)

| Model | Params (M) | GFLOPs | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---|---|---|---|---|---|---|
| YOLOv5s | 7.0 | 15.8 | 0.6310 | 0.5540 | 0.5860 | 0.2930 | 149.25 |
| YOLOv8n | 3.0 | 8.1 | 0.6181 | 0.5439 | 0.5712 | 0.3056 | 290.03 |
| **YOLOv8s** | 11.1 | 28.4 | **0.6598** | **0.5735** | **0.6170** | **0.3329** | 104.01 |

- **YOLOv8s** is the most accurate; **YOLOv8n** is the fastest (about 290 FPS).
- "Other corruption" is the easiest class; "pothole" is the hardest (and the rarest in training data).
- Trained on Japan only, YOLOv8s reaches mAP@0.5 = 0.5966 on Japan but drops to between 0.0673 and 0.4282 on the other five countries, which shows a strong domain shift.

## Repository structure

```
notebooks/       Kaggle notebooks, each with explanatory markdown sections
                   yolov5s.ipynb, yolov8n.ipynb, yolov8snew.ipynb   training (YOLOv8s is in yolov8snew)
                   comperision.ipynb                                 test-set comparison of the three models
                   crosscountryvalidation.ipynb                      Japan-only training and cross-country test
app/             Streamlit demo (app.py) and trained weights (road_damage_model.pt)
docs/            Project report (PDF), defence slides (PDF) and the project proposal
AI_USAGE.md      Disclosure of generative AI usage
```

## Run the Streamlit app

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

Run it from inside `app/`, because `app.py` loads `road_damage_model.pt` from the current folder. You can swap in another model by replacing that file.

## Dataset

The experiments use the Kaggle YOLO-format version of RDD2022 (26,869 train / 5,758 validation / 5,758 test images, 5 classes). **The dataset is not included in this repository**:

- Kaggle version: https://www.kaggle.com/datasets/aliabdelmenam/rdd-2022
- Original paper: Arya et al., *RDD2022: A multi-national image dataset for automatic road damage detection*, arXiv:2209.08538

## Notebooks

The notebooks were run on Kaggle with an NVIDIA Tesla T4 GPU. Paths inside them point to Kaggle's `/kaggle/input` and `/kaggle/working`, so adjust them if you run them elsewhere. Run the three training notebooks first; the comparison notebook then needs their `best.pt` files as inputs.

## Training settings (all models)

Image size 640×640, batch size 16, 50 epochs, early-stopping patience 15, seed 42, pre-trained weights.

## License

No license has been chosen yet. The dataset keeps its own terms.
