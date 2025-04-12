import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from matplotlib import cm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import models
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore


def extract_data_from_filename(filepath):
    parts = filepath.split("_")
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])

    return age, gender, race


def prepare_dataframe(dataset_dir):
    filepaths = []
    ages = []
    genders = []
    ethnicities = []
    age_classes = []
    
    BROKEN_FILES = [
        "39_1_20170116174525125.jpg.chip.jpg",
        "61_1_20170109142408075.jpg.chip.jpg",
        "61_1_20170109150557335.jpg.chip.jpg"
    ]

    for file in os.listdir(dataset_dir):
        if file.endswith(".jpg"):
            if file in BROKEN_FILES:
                continue

            filepath = os.path.join(dataset_dir, file)
            filepaths.append(filepath)
            age, gender, race = extract_data_from_filename(file)
            ages.append(age)
            genders.append(gender)
            ethnicities.append(race)
            age_classes.append(age // 10)

    return pd.DataFrame({
        "Image_Path": filepaths,
        "Age": ages,
        "Gender": genders,
        "Ethnicity": ethnicities,
        "Age_Class": age_classes
    })


def load_and_preprocess_image(path, label=None, img_height=200, img_width=200):
    try:
        image_string = tf.io.read_file(path)
        image = tf.io.decode_image(image_string, channels=3, expand_animations=False)
        image = tf.image.resize(image, [img_height, img_width])
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape([img_height, img_width, 3])

        if label is not None:
            return image, label
        else:
            return image
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        raise e


# Stop warning due to lack of conversion
@tf.autograph.experimental.do_not_convert
def create_dataset(paths, labels, batch_size, img_height=200, img_width=200, is_training=True):
    path_label_ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    image_label_ds = path_label_ds.map(
        lambda path, label: load_and_preprocess_image(path, label, img_height, img_width),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = image_label_ds.batch(batch_size, drop_remainder=is_training)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def evaluate_and_predict(model_path, model_name, test_dataset, true_labels_df):
    model = models.load_model(model_path)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    predictions_raw = model.predict(test_dataset, verbose=0)
    predicted_indices = np.argmax(predictions_raw, axis=1)
    predicted_confidence = np.max(predictions_raw, axis=1)

    true_labels_df["Predicted_Age_Class"] = predicted_indices
    true_labels_df["Confidence"] = predicted_confidence
    true_indices = true_labels_df["Age_Class"]

    num_classes = predictions_raw.shape[1]
    class_names = [f"{i*10}-{i*10+9}" for i in range(num_classes)]
    report_string = classification_report(true_indices, predicted_indices, target_names=class_names)
    print("Classification Report:")
    print(report_string)

    report_dict = classification_report(true_indices, predicted_indices, target_names=class_names, output_dict=True)

    return {
        "model_name": model_name,
        "true_labels": true_indices,
        "predicted_labels": predicted_indices,
        "predicted_probabilities": predictions_raw,
        "classification_report": report_dict
    }, true_labels_df, model


def prepare_comparison_data(report_dict_base, report_dict_balanced, support_threshold):
    df_base = pd.DataFrame(report_dict_base).transpose()
    df_balanced = pd.DataFrame(report_dict_balanced).transpose()

    summary_rows = ["accuracy", "macro avg", "weighted avg"]
    class_filter_base = ~df_base.index.isin(summary_rows)
    class_filter_balanced = ~df_balanced.index.isin(summary_rows)

    df_base_classes = df_base[class_filter_base].copy()
    df_balanced_classes = df_balanced[class_filter_balanced].copy()

    metrics_to_show = ["precision", "recall", "f1-score", "support"]
    valid_metrics = [m for m in metrics_to_show if m in df_base_classes.columns and m in df_balanced_classes.columns]

    df_base_subset = df_base_classes[valid_metrics].add_suffix("_Base")
    df_balanced_subset = df_balanced_classes[valid_metrics].add_suffix("_Balanced")

    df_full_comparison = pd.concat([df_base_subset, df_balanced_subset], axis=1)

    support_col_final_name = "support"
    df_full_comparison = df_full_comparison.rename(columns={"support_Base": support_col_final_name})
    df_full_comparison = df_full_comparison.drop(columns=["support_Balanced"])

    column_order = []
    for metric in ["precision", "recall", "f1-score"]:
        base_col = f"{metric}_Base"
        balanced_col = f"{metric}_Balanced"
        if base_col in df_full_comparison.columns and balanced_col in df_full_comparison.columns:
            column_order.extend([base_col, balanced_col])
    column_order.append(support_col_final_name)

    df_full_comparison = df_full_comparison[column_order]
    df_full_comparison.index = df_full_comparison.index.astype(str)
    df_full_comparison = df_full_comparison.sort_index(key=lambda x: x.str.split("-").str[0].astype(int))

    minority_classes = []
    df_full_comparison[support_col_final_name] = pd.to_numeric(df_full_comparison[support_col_final_name])
    minority_classes = df_full_comparison[df_full_comparison[support_col_final_name] < support_threshold].index.tolist()

    return df_full_comparison, minority_classes


def plot_comparison_metric(df_comp_all, metric_base_name, minority_classes_list):
    base_col = f"{metric_base_name}_Base"
    balanced_col = f"{metric_base_name}_Balanced"

    labels = df_comp_all.index
    base_values = df_comp_all[base_col]
    balanced_values = df_comp_all[balanced_col]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    rects_base = ax.bar(x - width/2, base_values, width, label="Base Model", color="skyblue")
    rects_class_weight = ax.bar(x + width/2, balanced_values, width, label="Class Weighting Model", color="lightcoral")

    for i, label in enumerate(labels):
        if label in minority_classes_list:
            rects_base[i].set_hatch("//")
            rects_class_weight[i].set_hatch("//")

    metric_title = metric_base_name.replace("-", " ").title()
    ax.set_ylabel(f"{metric_title} Score")
    ax.set_title(f"{metric_title} Comparison (Slashed = Minority Class)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar_label(rects_base, padding=3, fmt="%.3f", fontsize=8)
    ax.bar_label(rects_class_weight, padding=3, fmt="%.3f", fontsize=8)

    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(df, title, cmap="BuPu"):
    y_true = df["Age_Class"].astype(int)
    y_pred = df["Predicted_Age_Class"].astype(int)

    num_classes = 9
    labels_for_matrix = list(range(num_classes))

    class_names = [f"{i*10}-{i*10+9}" for i in labels_for_matrix]

    cm = confusion_matrix(y_true, y_pred, labels=labels_for_matrix, normalize="true")

    _, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    disp.plot(ax=ax, cmap=cmap, values_format=".2%")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def calculate_accuracy_by_gender_and_ethnicity(df):
    df_copy = df.copy()
    df_copy["is_correct"] = (df_copy["Age_Class"] == df_copy["Predicted_Age_Class"])

    grouped_data = df_copy.groupby(["Gender", "Ethnicity"])["is_correct"]
    accuracy = grouped_data.mean()
    support = grouped_data.size()

    results_df = pd.DataFrame({"Accuracy": accuracy, "Support": support})
    results_df = results_df[results_df["Support"] > 0]

    return results_df


def plot_accuracy_comparison_by_group(accuracy_base, accuracy_balanced):
    gender_map = {0: "Male", 1: "Female"}
    ethnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

    acc_base_combined = accuracy_base.rename(columns={"Accuracy": "Base Model"})
    acc_balanced_combined = accuracy_balanced.rename(columns={"Accuracy": "Class Weighting Model"})

    df_plot_combined = pd.concat([
        acc_base_combined["Base Model"],
        acc_balanced_combined["Class Weighting Model"],
        acc_base_combined["Support"]
    ], axis=1, join="outer")

    df_plot_combined = df_plot_combined.reset_index()
    df_plot_combined["Gender_Label"] = df_plot_combined["Gender"].map(gender_map)
    df_plot_combined["Ethnicity_Label"] = df_plot_combined["Ethnicity"].map(ethnicity_map)

    print("Combined Accuracy by Gender and Ethnicity:")
    print(df_plot_combined[["Gender_Label", "Ethnicity_Label", "Base Model", "Class Weighting Model", "Support"]].round(3))

    df_melted = pd.melt(
        df_plot_combined,
        id_vars=["Gender_Label", "Ethnicity_Label", "Support"],
        value_vars=["Base Model", "Class Weighting Model"],
        var_name="Model:",
        value_name="Accuracy"
    )

    g = sns.catplot(
        data=df_melted,
        x="Ethnicity_Label",
        y="Accuracy",
        hue="Model:",
        col="Gender_Label",
        kind="bar",
        palette=["skyblue", "lightcoral"],
        height=5,
        aspect=1.2,
        legend_out=True,
        order=list(ethnicity_map.values())
    )

    g.figure.suptitle("Age Prediction Accuracy by Ethnicity and Gender", y=1.03)
    g.set_axis_labels("Ethnicity", "Accuracy")
    g.set_titles("{col_name}")
    g.set(ylim=(0, 1.05))
    g.tick_params(axis="x")

    for ax in g.axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

    plt.show()


def generate_heatmaps_from_sample(gradcam, image_path, label, save_file=None, annotation_text=""):
    image = load_and_preprocess_image(image_path)
    image_numpy = image.numpy()

    input_image_display = np.uint8(image_numpy * 255)
    input_image = np.expand_dims(image_numpy, axis=0)
    age_group = f"{label * 10}-{label * 10 + 9}"
    
    plot_heatmap(gradcam, label, f"Heatmap for Predicted Label {age_group}", input_image, input_image_display, save_file, annotation_text)


def plot_heatmap(gradcam, label, title, input_image, input_image_display, save_file=None, annotation_text=""):
    cam = gradcam(CategoricalScore(label), input_image)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)

    _, (ax_img, ax_txt) = plt.subplots(2, 1, figsize=(6, 8), gridspec_kw={'height_ratios': [4, 1]})

    ax_img.imshow(input_image_display)
    ax_img.imshow(heatmap, alpha=0.5)
    ax_img.set_title(title)
    ax_img.axis("off")

    ax_txt.axis("off")
    if annotation_text:
        ax_txt.text(0.5, 0.5, annotation_text, fontsize=12, color="white", ha="center", va="center", bbox=dict(facecolor="black", alpha=0.7), transform=ax_txt.transAxes)

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight")
    plt.show()


# annotation_lines.append() + print() is very inefficient and should be combined, but I was lazy
def predict_xai(model, image_path, plot_heatmap=False, save_file=None):
    image = load_and_preprocess_image(image_path)
    image_numpy = image.numpy()    
    image_batch = np.expand_dims(image_numpy, axis=0)
    predictions = model.predict(image_batch, verbose=0)[0]

    sorted_indices = np.argsort(predictions)[::-1]
    top3 = [(sorted_indice, predictions[sorted_indice]) for sorted_indice in sorted_indices[:3]]

    # Option 1: If the top prediction has >=90% confidence
    if predictions[sorted_indices[0]] >= 0.90:
        chosen_indices = [sorted_indices[0]]
    else:
        # Option 2: Combine predictions (in sorted order) until cumulative confidence >= 90%.
        cumulative_conf = 0.0
        chosen_indices = []
        for sorted_indice in sorted_indices:
            cumulative_conf += predictions[sorted_indice]
            chosen_indices.append(sorted_indice)
            if cumulative_conf >= 0.90:
                break

    lower_bound = min(chosen_indices) * 10
    upper_bound = max(chosen_indices) * 10 + 9
    final_range = f"{lower_bound}-{upper_bound}"
    final_confidence = sum(predictions[chosen_indice] for chosen_indice in chosen_indices)
    
    annotation_lines = []

    print("Top 3 Predictions:")
    annotation_lines.append("Top 3 Predictions:")
    for prediction, conf in top3:
        age_range = f"{prediction * 10}-{prediction * 10 + 9}"
        print(f"Age Range {age_range}: {conf * 100:.2f}% ({conf:.7f})")
        annotation_lines.append(f"Age Range {age_range}: {conf * 100:.2f}% ({conf:.7f})")
    
    print("")
    print("Final Age Group Prediction:")
    print(f"Age Group: {final_range}")
    print(f"Cumulative Confidence: {final_confidence * 100:.2f}% ({final_confidence:.7f})")

    annotation_lines.append("")
    annotation_lines.append("Final Age Group Prediction:")
    annotation_lines.append(f"Age Group: {final_range}")
    annotation_lines.append(f"Cumulative Confidence: {final_confidence * 100:.2f}% ({final_confidence:.7f})")

    annotation_text = "\n".join(annotation_lines)

    if plot_heatmap:
        top_prediction = np.argmax(predictions)
        gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=True)
        generate_heatmaps_from_sample(gradcam, image_path, top_prediction, save_file, annotation_text)


def find_prototype_and_criticism(df, gender, ethnicity, age_class, csv_filename=None, save_to_csv=False):
    gender_map = {0: "Male", 1: "Female"}
    ethnicity_map = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others"}

    subset = df[
        (df["Gender"] == gender) &
        (df["Ethnicity"] == ethnicity) &
        (df["Age_Class"] == age_class)
    ].copy()
    
    subset["diff"] = (subset["Predicted_Age_Class"] - subset["Age_Class"])\
    
    if subset.empty:
        print(f"No records found for Gender={gender}, Ethnicity={ethnicity}, Age_Class={age_class} in Test Set")
        return None
    
    prototype_candidates = subset[subset["diff"] == 0]
    prototype_candidates = prototype_candidates.sort_values(by="Confidence", ascending=False)

    if not prototype_candidates.empty:
        prototype = prototype_candidates.iloc[0]
    else:
        prototype = subset.sort_values("diff").iloc[0]

    criticism = subset.sort_values("diff", ascending=False).iloc[0]

    real_age_group = f"{age_class * 10}-{age_class * 10 + 9}"
    predicted_prototype_age_group = f"{prototype['Predicted_Age_Class'] * 10}-{prototype['Predicted_Age_Class'] * 10 + 9}"
    predicted_criticism_age_group = f"{criticism['Predicted_Age_Class'] * 10}-{criticism['Predicted_Age_Class'] * 10 + 9}"

    print("INPUT PARAMETERS:")
    print(f"Gender: {gender} ({gender_map[gender]}), Ethnicity: {ethnicity} ({ethnicity_map[ethnicity]}), Age_Class (Ground Truth): {age_class} ({real_age_group})")
    print("")
    print("PROTOTYPE SAMPLE:")
    print(f"Image Path: {prototype['Image_Path']}")
    print(f"Age: {prototype['Age']}")
    print(f"Age_Class: {real_age_group}")
    print(f"Predicted_Age_Class: {predicted_prototype_age_group} (Diff: {prototype['diff']})")
    print("")
    print("CRITICISM SAMPLE:")
    print(f"Image Path: {criticism['Image_Path']}")
    print(f"Age: {criticism['Age']}")
    print(f"Age_Class: {real_age_group}")
    print(f"Predicted_Age_Class: {predicted_criticism_age_group} (Diff: {criticism['diff']})")

    result = {
        "Gender": gender,
        "Gender_Label": gender_map[gender],
        "Ethnicity": ethnicity,
        "Ethnicity_Label": ethnicity_map[ethnicity],
        "Ground_Truth_Age_Class": real_age_group,
        "Prototype_Image_Path": prototype["Image_Path"],
        "Prototype_Age": prototype["Age"],
        "Prototype_Predicted_Age_Class": predicted_prototype_age_group,
        "Prototype_Diff": prototype["diff"],
        "Criticism_Image_Path": criticism["Image_Path"],
        "Criticism_Age": criticism["Age"],
        "Criticism_Predicted_Age_Class": predicted_criticism_age_group,
        "Criticism_Diff": criticism["diff"]
    }
    
    if save_to_csv:
        if csv_filename is not None:
            result_df = pd.DataFrame([result])
            if os.path.exists(csv_filename):
                result_df.to_csv(csv_filename, mode="a", header=False, index=False)
            else:
                result_df.to_csv(csv_filename, index=False)
    
    return result