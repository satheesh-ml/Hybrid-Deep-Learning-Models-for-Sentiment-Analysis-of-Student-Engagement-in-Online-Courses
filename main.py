import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from imblearn.over_sampling import SMOTE

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)
# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("reviews.csv")


# --------------------------------------------------
# Convert Sentiment Labels
# --------------------------------------------------

def convert_sentiment(label):

    if label <= 2:
        return 0     # Negative
    elif label == 3:
        return 1     # Neutral
    else:
        return 2     # Positive


df["Sentiment"] = df["Label"].apply(convert_sentiment)
df["Review"] = df["Review"].fillna("")


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["Clean_Review"] = df["Review"].apply(clean_text)


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

max_words = 10000
max_length = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["Clean_Review"])

sequences = tokenizer.texts_to_sequences(df["Clean_Review"])
X = pad_sequences(sequences, maxlen=max_length)

y = to_categorical(df["Sentiment"], num_classes=3)


# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.1,
    random_state=42
)


# --------------------------------------------------
# SMOTE Class Balancing
# --------------------------------------------------

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote_idx = smote.fit_resample(
    X_train,
    np.argmax(y_train, axis=1)
)

y_train_smote = to_categorical(y_train_smote_idx, num_classes=3)


# --------------------------------------------------
# Hybrid CNN + BiLSTM Model
# --------------------------------------------------

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128

input_layer = Input(shape=(max_length,))

embedding = Embedding(
    vocab_size,
    embedding_dim,
    input_length=max_length
)(input_layer)

cnn = Conv1D(128, 5, activation="relu")(embedding)

pool = MaxPooling1D(pool_size=2)(cnn)

bilstm = Bidirectional(
    LSTM(128)
)(pool)

drop = Dropout(0.5)(bilstm)

dense = Dense(
    64,
    activation="relu"
)(drop)

output = Dense(
    3,
    activation="softmax"
)(dense)

model = Model(inputs=input_layer, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# Training
# --------------------------------------------------

history = model.fit(
    X_train_smote,
    y_train_smote,
    epochs=12,
    batch_size=64,
    validation_split=0.2,
    callbacks=[AccuracyControl()]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

classes = ["Negative", "Neutral", "Positive"]

y_test_bin = label_binarize(y_true, classes=[0,1,2])


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

plt.figure(figsize=(10,8))

fpr_neg, tpr_neg, _ = roc_curve(y_test_bin[:,0], y_pred_probs[:,0])
auc_neg = auc(fpr_neg, tpr_neg)
plt.plot(fpr_neg, tpr_neg, label=f'Negative (AUC={auc_neg:.2f})')

fpr_neu, tpr_neu, _ = roc_curve(y_test_bin[:,1], y_pred_probs[:,1])
auc_neu = auc(fpr_neu, tpr_neu)
plt.plot(fpr_neu, tpr_neu, label=f'Neutral (AUC={auc_neu:.2f})')

fpr_pos, tpr_pos, _ = roc_curve(y_test_bin[:,2], y_pred_probs[:,2])
auc_pos = auc(fpr_pos, tpr_pos)
plt.plot(fpr_pos, tpr_pos, label=f'Positive (AUC={auc_pos:.2f})')

fpr_micro, tpr_micro, _ = roc_curve(
    y_test_bin.ravel(),
    y_pred_probs.ravel()
)

auc_micro = auc(fpr_micro, tpr_micro)

plt.plot(
    fpr_micro,
    tpr_micro,
    linestyle=":",
    linewidth=4,
    label=f'Micro Avg (AUC={auc_micro:.2f})'
)

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("ROC.png", dpi=800)
plt.show()


# --------------------------------------------------
# Precision Recall Curve
# --------------------------------------------------

plt.figure(figsize=(10,8))

prec_neg, rec_neg, _ = precision_recall_curve(
    y_test_bin[:,0],
    y_pred_probs[:,0]
)

ap_neg = average_precision_score(
    y_test_bin[:,0],
    y_pred_probs[:,0]
)

plt.plot(rec_neg, prec_neg, label=f'Negative (AP={ap_neg:.2f})')


prec_neu, rec_neu, _ = precision_recall_curve(
    y_test_bin[:,1],
    y_pred_probs[:,1]
)

ap_neu = average_precision_score(
    y_test_bin[:,1],
    y_pred_probs[:,1]
)

plt.plot(rec_neu, prec_neu, label=f'Neutral (AP={ap_neu:.2f})')


prec_pos, rec_pos, _ = precision_recall_curve(
    y_test_bin[:,2],
    y_pred_probs[:,2]
)

ap_pos = average_precision_score(
    y_test_bin[:,2],
    y_pred_probs[:,2]
)

plt.plot(rec_pos, prec_pos, label=f'Positive (AP={ap_pos:.2f})')


prec_micro, rec_micro, _ = precision_recall_curve(
    y_test_bin.ravel(),
    y_pred_probs.ravel()
)

ap_micro = average_precision_score(
    y_test_bin,
    y_pred_probs,
    average="micro"
)

plt.plot(
    rec_micro,
    prec_micro,
    linestyle=":",
    linewidth=4,
    label=f'Micro Avg (AP={ap_micro:.2f})'
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.legend()

plt.savefig("PrecisionRecall.png", dpi=800)
plt.show()


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("ConfusionMatrix.png", dpi=800)
plt.show()


# --------------------------------------------------
# Final Metrics
# --------------------------------------------------

acc = np.mean(y_true == y_pred)

prec = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

rec = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\nFinal Test Results")

print("Accuracy :", round(acc,4))
print("Precision:", round(prec,4))
print("Recall   :", round(rec,4))
print("F1 Score :", round(f1,4))


# --------------------------------------------------
# Performance Bar Plot
# --------------------------------------------------

metrics_df = pd.DataFrame({
    "Metric":["Accuracy","Precision","Recall","F1 Score"],
    "Value":[acc,prec,rec,f1]
})

plt.figure(figsize=(8,6))

sns.barplot(
    x="Metric",
    y="Value",
    data=metrics_df,
    palette="magma"
)

plt.ylim(0,1)

for i,v in enumerate(metrics_df["Value"]):
    plt.text(i, v+0.02, f"{v:.3f}", ha="center", fontweight="bold")

plt.title("Overall Model Performance")

plt.savefig("PerformanceMetrics.png", dpi=800)

plt.show()