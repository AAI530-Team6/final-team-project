Activity Recognition

What it is
A production minded Human Activity Recognition system built on PAMAP2 wearable sensor time series data. It classifies activities from multi sensor streams using sliding windows, compares classical ML vs 1D CNN, and exports time aligned predictions for a Tableau dashboard.

Why it matters

    Wearable monitoring needs models that generalize to new people. This project uses subject level splits and GroupKFold validation to avoid leakage and report realistic performance.

How it works

    Windowing: 10 second windows with 50 percent overlap and no label crossing

    Features: time domain and frequency domain window features

    Models: ExtraTrees, Random Forest, tuned LinearSVC, plus a PyTorch 1D CNN

    Evaluation: Macro F1, Balanced Accuracy, confusion matrix analysis on unseen subjects

    Output: window predictions projected to time aligned records for visualization

Results

Best classical models reach about 0.95 Macro F1 and about 0.94 Balanced Accuracy on unseen subjects.
The 1D CNN reaches about 0.93 Macro F1 on the same split.

Repo layout

    notebooks: EDA, feature extraction, training, evaluation

    artifacts: saved model and feature column list

    build_tableau_csv.py: generates Tableau ready CSV outputs

Quick start

1 Install dependencies
2 Run the notebook to train models and export artifacts
3 Generate Tableau CSV with the export script
4 Build the dashboard in Tableau Public