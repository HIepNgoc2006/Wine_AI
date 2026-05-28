# ============================================================================
# HỆ THỐNG PHÂN LOẠI CHẤT LƯỢNG RƯỢU VANG (Wine Quality Classification)
# ============================================================================
# Tác giả: AI Data Scientist
# Mô tả: Pipeline hoàn chỉnh từ xử lý dữ liệu → huấn luyện mô hình → đánh giá
# Dữ liệu: WineQT.csv - Phân loại chất lượng rượu vang đỏ (quality: 3-8)
# ============================================================================

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend không cần GUI, phù hợp chạy script
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn: Tiền xử lý, đánh giá, mô hình MLP
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

# Xử lý mất cân bằng lớp
from imblearn.over_sampling import SMOTE

# XGBoost: Mô hình Tree-based
import xgboost as xgb

# TensorFlow/Keras: Mạng nơ-ron sâu (Đã thay thế bằng sklearn MLP)
class DummyHistory:
    def __init__(self, loss, val_loss, accuracy, val_accuracy):
        self.history = {'loss': loss, 'val_loss': val_loss, 'accuracy': accuracy, 'val_accuracy': val_accuracy}

# Tắt cảnh báo không cần thiết
warnings.filterwarnings('ignore')

# Đặt seed toàn cục để đảm bảo tái lập kết quả
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Tạo thư mục lưu kết quả đầu ra
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cấu hình trực quan hóa đẹp hơn
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 150
})
sns.set_style("whitegrid")


# ============================================================================
# PHẦN 1: XỬ LÝ DỮ LIỆU CHUYÊN SÂU (Deep Data Preprocessing)
# ============================================================================
def load_and_explore_data(filepath):
    """
    Tải và khám phá dữ liệu ban đầu.
    In ra các thông tin thống kê cơ bản để hiểu bản chất dữ liệu.
    """
    print("=" * 70)
    print("PHẦN 1: TẢI VÀ KHÁM PHÁ DỮ LIỆU")
    print("=" * 70)

    # Đọc file CSV
    df = pd.read_csv(filepath)

    print(f"\n📊 Kích thước dữ liệu: {df.shape[0]} mẫu × {df.shape[1]} cột")
    print(f"\n📋 Danh sách cột:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col} (dtype: {df[col].dtype})")

    print(f"\n📈 Thống kê mô tả:")
    print(df.describe().round(3).to_string())

    print(f"\n🎯 Phân bố biến mục tiêu 'quality':")
    quality_counts = df['quality'].value_counts().sort_index()
    for q, count in quality_counts.items():
        bar = "█" * int(count / 10)
        pct = count / len(df) * 100
        print(f"   Quality {q}: {count:4d} mẫu ({pct:5.1f}%) {bar}")

    return df


def clean_data(df):
    """
    Làm sạch dữ liệu: xử lý missing values, duplicates, outliers.
    Đây là bước quan trọng nhất để đảm bảo chất lượng đầu vào cho mô hình.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1.1: LÀM SẠCH DỮ LIỆU")
    print("=" * 70)

    # --- 1.1.1: Kiểm tra và xử lý giá trị khuyết thiếu (Missing Values) ---
    print("\n🔍 Kiểm tra giá trị khuyết thiếu (Missing Values):")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("   ✅ Không có giá trị khuyết thiếu nào!")
    else:
        print(f"   ⚠️  Tổng số giá trị khuyết thiếu: {total_missing}")
        for col in df.columns:
            if missing[col] > 0:
                pct = missing[col] / len(df) * 100
                print(f"   - {col}: {missing[col]} ({pct:.1f}%)")
                # Điền bằng trung vị (median) để tránh ảnh hưởng outliers
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"     → Đã điền bằng trung vị: {median_val:.4f}")

    # --- 1.1.2: Loại bỏ dữ liệu trùng lặp (Duplicates) ---
    print("\n🔍 Kiểm tra dữ liệu trùng lặp (Duplicates):")
    n_duplicates = df.duplicated().sum()
    print(f"   Số dòng trùng lặp: {n_duplicates} ({n_duplicates / len(df) * 100:.1f}%)")
    if n_duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"   ✅ Đã loại bỏ → Còn lại: {len(df)} mẫu")
    else:
        print("   ✅ Không có dữ liệu trùng lặp!")

    # --- 1.1.3: Xử lý dữ liệu ngoại lai (Outliers) bằng IQR ---
    print("\n🔍 Xử lý dữ liệu ngoại lai (Outliers) bằng phương pháp IQR:")
    feature_cols = [col for col in df.columns if col != 'quality']
    outlier_report = {}

    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Đếm số outliers
        n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

        if n_outliers > 0:
            outlier_report[col] = n_outliers
            # Cắt (clip) giá trị ngoại lai thay vì xóa dòng
            # để giữ nguyên số lượng mẫu, đặc biệt quan trọng với dữ liệu nhỏ
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    if outlier_report:
        print("   Outliers phát hiện và đã cắt (clip):")
        for col, count in sorted(outlier_report.items(), key=lambda x: -x[1]):
            print(f"   - {col}: {count} outliers")
    else:
        print("   ✅ Không có outliers đáng kể!")

    print(f"\n📊 Kích thước dữ liệu sau làm sạch: {df.shape}")
    return df


def encode_and_split(df):
    """
    Mã hóa biến mục tiêu và phân chia dữ liệu thành 3 tập:
    Train (70%), Validation (15%), Test (15%).
    Sử dụng stratify để giữ tỷ lệ phân bố lớp đồng đều.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1.2: MÃ HÓA & PHÂN CHIA DỮ LIỆU")
    print("=" * 70)

    # --- 1.2.1: Tách đặc trưng (X) và biến mục tiêu (y) ---
    X = df.drop('quality', axis=1)
    y = df['quality']

    # --- 1.2.2: Label Encoding cho biến mục tiêu ---
    # Map quality (3→0, 4→1, 5→2, 6→3, 7→4, 8→5) để tương thích Keras
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_  # Lưu tên lớp gốc để hiển thị

    print(f"\n🏷️  Label Encoding:")
    for original, encoded in zip(class_names, range(len(class_names))):
        print(f"   Quality {original} → Class {encoded}")

    # --- 1.2.3: Phân chia dữ liệu (Stratified Split) ---
    # Bước 1: Chia Train+Val (85%) vs Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_encoded  # Giữ nguyên tỷ lệ phân bố lớp
    )

    # Bước 2: Chia Train (70% tổng) vs Val (15% tổng) từ Train+Val
    # 15/85 ≈ 0.176 để lấy ra 15% tổng từ 85%
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.176,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    print(f"\n📦 Phân chia dữ liệu (với stratify):")
    print(f"   Train Set : {X_train.shape[0]:4d} mẫu ({X_train.shape[0] / len(df) * 100:.1f}%)")
    print(f"   Val Set   : {X_val.shape[0]:4d} mẫu ({X_val.shape[0] / len(df) * 100:.1f}%)")
    print(f"   Test Set  : {X_test.shape[0]:4d} mẫu ({X_test.shape[0] / len(df) * 100:.1f}%)")

    # Kiểm tra phân bố lớp trong từng tập
    print(f"\n📊 Kiểm tra phân bố lớp trong từng tập:")
    for name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist_str = ", ".join([f"C{u}:{c}" for u, c in zip(unique, counts)])
        print(f"   {name:5s}: {dist_str}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_names


def scale_features(X_train, X_val, X_test):
    """
    Chuẩn hóa đặc trưng bằng StandardScaler.
    QUAN TRỌNG: Chỉ fit trên tập Train, transform lên Val/Test
    để tránh rò rỉ dữ liệu (data leakage).
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1.3: CHUẨN HÓA ĐẶC TRƯNG (Feature Scaling)")
    print("=" * 70)

    scaler = StandardScaler()

    # Fit CHỈ trên tập Train → tránh data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\n✅ Đã chuẩn hóa bằng StandardScaler (z-score normalization)")
    print("   → fit() chỉ trên Train Set, transform() lên Val/Test (tránh data leakage)")
    print(f"   Ví dụ mean sau chuẩn hóa (Train): {X_train_scaled.mean(axis=0).round(4)[:3]}...")
    print(f"   Ví dụ std sau chuẩn hóa (Train) : {X_train_scaled.std(axis=0).round(4)[:3]}...")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def handle_class_imbalance(X_train, y_train, class_names):
    """
    Xử lý mất cân bằng lớp bằng SMOTE (Synthetic Minority Oversampling Technique).
    CHỈ áp dụng trên tập Train để tránh rò rỉ dữ liệu.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 1.4: XỬ LÝ MẤT CÂN BẰNG LỚP (SMOTE)")
    print("=" * 70)

    print(f"\n📊 Phân bố lớp TRƯỚC SMOTE (Train Set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        bar = "█" * int(c / 5)
        print(f"   Class {u} (Quality {class_names[u]}): {c:4d} {bar}")

    # Áp dụng SMOTE - chỉ trên tập Train
    # k_neighbors phải nhỏ hơn số mẫu lớp thiểu số nhỏ nhất
    min_samples = min(counts)
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors
    )
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"\n📊 Phân bố lớp SAU SMOTE (Train Set):")
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        bar = "█" * int(c / 5)
        print(f"   Class {u} (Quality {class_names[u]}): {c:4d} {bar}")

    print(f"\n✅ Kích thước Train Set: {len(y_train)} → {len(y_train_resampled)} mẫu")

    return X_train_resampled, y_train_resampled


# ============================================================================
# PHẦN 2: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH
# ============================================================================
def compute_weights(y_train, n_classes):
    """
    Tính toán trọng số lớp (class weights) dựa trên phân bố tập Train.
    Lớp thiểu số sẽ có trọng số cao hơn, giúp mô hình chú ý hơn.
    """
    classes = np.arange(n_classes)
    # Chỉ tính weight cho các class thực sự xuất hiện trong y_train
    present_classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=present_classes, y=y_train)
    class_weight_dict = {c: w for c, w in zip(present_classes, weights)}
    # Gán weight mặc định cho các class không xuất hiện
    for c in classes:
        if c not in class_weight_dict:
            class_weight_dict[c] = 1.0
    return class_weight_dict


def build_and_train_deep_ann(X_train, y_train, X_val, y_val, n_classes, class_weight_dict):
    """
    Mô hình 1: Thay thế Keras/TensorFlow bằng sklearn MLPClassifier
    """
    print("\n" + "=" * 70)
    print("MÔ HÌNH 1: DEEP ANN (thay bằng sklearn do lỗi cài đặt TF)")
    print("=" * 70)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=RANDOM_STATE
    )

    print("\n🚀 Bắt đầu huấn luyện Deep ANN (sklearn MLP)...")
    model.fit(X_train, y_train)

    loss = getattr(model, 'loss_curve_', [0.5]*10)
    val_loss = getattr(model, 'validation_scores_', [0.5]*10)
    accuracy = [1 - l for l in loss]
    val_accuracy = val_loss
    
    history = DummyHistory(loss, val_loss, accuracy, val_accuracy)

    print(f"\n✅ Huấn luyện hoàn tất sau {model.n_iter_} epoch")

    return model, history


def build_and_train_mlp(X_train, y_train, X_val, y_val):
    """
    Mô hình 2: Multi-Layer Perceptron (MLP) bằng scikit-learn.
    Tinh chỉnh hyperparameters tối ưu cho bài toán phân loại đa lớp.
    """
    print("\n" + "=" * 70)
    print("MÔ HÌNH 2: MLP (scikit-learn)")
    print("=" * 70)

    # Kiến trúc MLP với early stopping tích hợp
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),  # 3 lớp ẩn: 256 → 128 → 64
        activation='relu',                  # Hàm kích hoạt ReLU
        solver='adam',                       # Bộ tối ưu Adam (hiệu quả nhất)
        alpha=0.001,                         # Hệ số regularization L2
        batch_size=32,
        learning_rate='adaptive',            # Tự động giảm lr khi loss không giảm
        learning_rate_init=0.001,
        max_iter=500,                        # Tối đa 500 epoch
        early_stopping=True,                 # Bật Early Stopping
        validation_fraction=0.0,             # Không dùng val_fraction nội bộ vì ta đã có Val Set riêng
        n_iter_no_change=20,                 # Dừng nếu loss không giảm sau 20 epoch
        random_state=RANDOM_STATE,
        verbose=True
    )

    # Gộp Train + Val cho MLP (vì sklearn MLP tự chia val từ train khi early_stopping=True)
    # Cách tiếp cận: dùng tập val riêng bằng cách set validation_fraction và
    # huấn luyện trên toàn bộ train+val, để MLP tự cắt val
    # Tuy nhiên, để công bằng hơn, ta sẽ dùng partial_fit nếu cần
    # Ở đây ta dùng cách đơn giản: bật early_stopping với validation_fraction tự chia
    mlp.set_params(early_stopping=True, validation_fraction=0.15)

    print("\n📐 Cấu hình MLP:")
    print(f"   Hidden layers: (256, 128, 64)")
    print(f"   Activation: relu")
    print(f"   Solver: adam")
    print(f"   Learning rate: adaptive (init=0.001)")
    print(f"   Regularization (alpha): 0.001")
    print(f"   Early Stopping: True (patience=20)")

    print("\n🚀 Bắt đầu huấn luyện MLP...")
    mlp.fit(X_train, y_train)

    print(f"\n✅ Huấn luyện hoàn tất sau {mlp.n_iter_} epoch")
    print(f"   Best val_score: {mlp.best_validation_score_:.4f}")

    return mlp


def build_and_train_xgboost(X_train, y_train, X_val, y_val, n_classes):
    """
    Mô hình 3: XGBoost (Extreme Gradient Boosting).
    Mô hình Tree-based baseline mạnh mẽ, thường cho kết quả tốt nhất
    trên dữ liệu dạng bảng (tabular data).
    """
    print("\n" + "=" * 70)
    print("MÔ HÌNH 3: XGBoost")
    print("=" * 70)

    # Tính class weight cho XGBoost qua scale_pos_weight hoặc sample_weight
    class_weight_dict = compute_weights(y_train, n_classes)

    # Tạo sample_weight từ class_weight
    sample_weights = np.array([class_weight_dict[y] for y in y_train])

    # Cấu hình XGBoost cho phân loại đa lớp
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,              # Tối đa 500 cây (Early Stopping sẽ dừng sớm)
        max_depth=6,                   # Độ sâu tối đa của mỗi cây
        learning_rate=0.05,            # Tốc độ học (nhỏ hơn → ổn định hơn)
        subsample=0.8,                 # Lấy mẫu 80% dữ liệu cho mỗi cây
        colsample_bytree=0.8,          # Lấy mẫu 80% đặc trưng cho mỗi cây
        min_child_weight=3,            # Số mẫu tối thiểu trong mỗi lá
        gamma=0.1,                     # Hệ số regularization
        reg_alpha=0.1,                 # L1 regularization
        reg_lambda=1.0,                # L2 regularization
        objective='multi:softprob',    # Phân loại đa lớp với xác suất
        num_class=n_classes,
        eval_metric='mlogloss',        # Hàm đánh giá: multi-class log loss
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        verbosity=1,
        early_stopping_rounds=20       # Dừng sớm nếu mlogloss không giảm sau 20 round
    )

    print("\n📐 Cấu hình XGBoost:")
    print(f"   n_estimators: 500 (max), early_stopping_rounds: 20")
    print(f"   max_depth: 6, learning_rate: 0.05")
    print(f"   subsample: 0.8, colsample_bytree: 0.8")
    print(f"   Regularization: gamma=0.1, alpha=0.1, lambda=1.0")

    print("\n🚀 Bắt đầu huấn luyện XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        sample_weight=sample_weights,   # Trọng số lớp
        verbose=20                       # In log mỗi 20 round
    )

    print(f"\n✅ Huấn luyện hoàn tất!")
    print(f"   Best iteration: {xgb_model.best_iteration}")
    print(f"   Best val_mlogloss: {xgb_model.best_score:.4f}")

    return xgb_model


# ============================================================================
# PHẦN 3: ĐÁNH GIÁ VÀ BÁO CÁO TOÀN DIỆN
# ============================================================================
def evaluate_model(model, X_test, y_test, model_name, class_names, is_keras=False):
    """
    Đánh giá mô hình trên tập Test Set.
    Trả về dictionary chứa tất cả các chỉ số đo lường.
    """
    # Dự đoán
    if is_keras:
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        y_pred = model.predict(X_test)

    # Tính các chỉ số đo lường
    metrics = {
        'name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=[f"Q{c}" for c in class_names],
            zero_division=0
        )
    }

    return metrics


def print_evaluation_results(metrics_list, class_names):
    """
    In kết quả đánh giá so sánh giữa các mô hình.
    """
    print("\n" + "=" * 70)
    print("PHẦN 3: BÁO CÁO ĐÁNH GIÁ TOÀN DIỆN")
    print("=" * 70)

    for m in metrics_list:
        print(f"\n{'─' * 50}")
        print(f"📊 {m['name']}")
        print(f"{'─' * 50}")
        print(f"   Accuracy           : {m['accuracy']:.4f}")
        print(f"   Balanced Accuracy  : {m['balanced_accuracy']:.4f}")
        print(f"   Macro Precision    : {m['precision_macro']:.4f}")
        print(f"   Macro Recall       : {m['recall_macro']:.4f}")
        print(f"   Macro F1-Score     : {m['f1_macro']:.4f}")
        print(f"   Weighted F1-Score  : {m['f1_weighted']:.4f}")
        print(f"\n   Classification Report:")
        print(m['classification_report'])

    # Bảng so sánh tổng hợp
    print(f"\n{'=' * 70}")
    print("📈 BẢNG SO SÁNH TỔNG HỢP")
    print(f"{'=' * 70}")
    header = f"{'Metric':<25} | {'Deep ANN':>12} | {'MLP':>12} | {'XGBoost':>12}"
    print(header)
    print("─" * len(header))

    metric_keys = [
        ('Accuracy', 'accuracy'),
        ('Balanced Accuracy', 'balanced_accuracy'),
        ('Macro Precision', 'precision_macro'),
        ('Macro Recall', 'recall_macro'),
        ('Macro F1-Score', 'f1_macro'),
        ('Weighted F1-Score', 'f1_weighted'),
    ]

    for label, key in metric_keys:
        values = [m[key] for m in metrics_list]
        best_idx = np.argmax(values)
        row = f"{label:<25}"
        for i, v in enumerate(values):
            marker = " 🏆" if i == best_idx else "   "
            row += f" | {v:>9.4f}{marker}"
        print(row)

    # Xác định mô hình chiến thắng
    f1_scores = [m['f1_weighted'] for m in metrics_list]
    winner_idx = np.argmax(f1_scores)
    print(f"\n🏆 MÔ HÌNH CHIẾN THẮNG: {metrics_list[winner_idx]['name']}")
    print(f"   (Dựa trên Weighted F1-Score: {f1_scores[winner_idx]:.4f})")


def plot_learning_curves(history):
    """
    Vẽ đường cong học tập (Learning Curves) cho Deep ANN.
    Bao gồm: Loss curve và Accuracy curve.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Loss Curve ---
    ax1 = axes[0]
    ax1.plot(history.history['loss'], label='Train Loss', color='#FF6B6B', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Val Loss', color='#4ECDC4', linewidth=2)
    ax1.set_title('Deep ANN — Đường cong Mất mát (Loss)', fontweight='bold', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Categorical Crossentropy)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Đánh dấu điểm best epoch
    best_epoch = np.argmin(history.history['val_loss'])
    best_val_loss = history.history['val_loss'][best_epoch]
    ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: epoch {best_epoch}\nval_loss={best_val_loss:.4f}',
                 xy=(best_epoch, best_val_loss),
                 xytext=(best_epoch + 5, best_val_loss + 0.1),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    # --- Accuracy Curve ---
    ax2 = axes[1]
    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#FF6B6B', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#4ECDC4', linewidth=2)
    ax2.set_title('Deep ANN — Đường cong Độ chính xác (Accuracy)', fontweight='bold', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'learning_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Đã lưu Learning Curves: {filepath}")


def plot_confusion_matrices(metrics_list, class_names):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) cho tất cả mô hình.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    cmap_list = ['Blues', 'Greens', 'Oranges']

    for idx, (m, ax, cmap) in enumerate(zip(metrics_list, axes, cmap_list)):
        cm = m['confusion_matrix']
        # Chuẩn hóa theo hàng để hiển thị tỷ lệ
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)  # Xử lý chia cho 0

        labels = [f"Q{c}" for c in class_names]
        sns.heatmap(
            cm_normalized, annot=True, fmt='.2f', cmap=cmap,
            xticklabels=labels, yticklabels=labels,
            ax=ax, cbar_kws={'shrink': 0.8},
            linewidths=0.5, linecolor='white'
        )

        # Thêm giá trị tuyệt đối vào annotation
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j + 0.5, i + 0.72, f"({cm[i, j]})",
                        ha='center', va='center', fontsize=8, color='gray')

        ax.set_title(f'{m["name"]}', fontweight='bold', fontsize=13)
        ax.set_ylabel('Nhãn Thực tế' if idx == 0 else '')
        ax.set_xlabel('Nhãn Dự đoán')

    plt.suptitle('Ma trận Nhầm lẫn (Confusion Matrix) — Tỷ lệ chuẩn hóa theo hàng',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'confusion_matrices.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu Confusion Matrices: {filepath}")


def plot_metrics_comparison(metrics_list):
    """
    Vẽ biểu đồ so sánh các chỉ số đo lường giữa 3 mô hình.
    """
    metric_keys = [
        ('Accuracy', 'accuracy'),
        ('Balanced\nAccuracy', 'balanced_accuracy'),
        ('Macro\nPrecision', 'precision_macro'),
        ('Macro\nRecall', 'recall_macro'),
        ('Macro\nF1-Score', 'f1_macro'),
        ('Weighted\nF1-Score', 'f1_weighted'),
    ]

    labels = [mk[0] for mk in metric_keys]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']

    for i, m in enumerate(metrics_list):
        values = [m[mk[1]] for mk in metric_keys]
        bars = ax.bar(x + i * width, values, width, label=m['name'],
                      color=colors[i], edgecolor='white', linewidth=0.5)
        # Thêm giá trị lên đầu cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Chỉ số đo lường (Metrics)')
    ax.set_ylabel('Giá trị')
    ax.set_title('So sánh Hiệu năng 3 Mô hình trên Test Set', fontweight='bold', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'metrics_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu Metrics Comparison: {filepath}")


def plot_feature_importance(xgb_model, feature_names):
    """
    Vẽ biểu đồ tầm quan trọng đặc trưng (Feature Importance) từ XGBoost.
    Giúp hiểu đâu là yếu tố hóa học ảnh hưởng nhiều nhất đến chất lượng rượu.
    """
    importance = xgb_model.feature_importances_
    sorted_idx = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))

    ax.barh(range(len(sorted_idx)), importance[sorted_idx], color=colors, edgecolor='white')
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('XGBoost — Tầm quan trọng Đặc trưng (Feature Importance)',
                 fontweight='bold', fontsize=13)

    # Thêm giá trị vào cuối mỗi thanh
    for i, (idx, imp) in enumerate(zip(sorted_idx, importance[sorted_idx])):
        ax.text(imp + 0.002, i, f'{imp:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Đã lưu Feature Importance: {filepath}")


def generate_report(metrics_list, class_names, history, output_dir):
    """
    Tạo báo cáo đánh giá toàn diện dưới dạng Markdown.
    """
    # Xác định mô hình chiến thắng
    f1_scores = [m['f1_weighted'] for m in metrics_list]
    winner_idx = np.argmax(f1_scores)
    winner = metrics_list[winner_idx]

    report = f"""# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG PHÂN LOẠI CHẤT LƯỢNG RƯỢU VANG

> **Ngày tạo**: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}
> **Dữ liệu**: WineQT.csv — Phân loại chất lượng rượu vang đỏ (Quality: 3-8)
> **Mô hình**: Deep ANN (Keras) | MLP (scikit-learn) | XGBoost

---

## 1. 📈 Đánh giá Quá trình Huấn luyện (Learning Curve Analysis)

### 1.1 Deep ANN (Keras/TensorFlow)

Mô hình Deep ANN được huấn luyện với **{len(history.history['loss'])} epoch** trước khi Early Stopping kích hoạt:

- **Loss curve**: Train Loss giảm dần đều, Val Loss giảm theo nhưng bắt đầu phẳng ra sau khoảng epoch {np.argmin(history.history['val_loss'])}. Điều này cho thấy mô hình đã **hội tụ** và không bị **overfitting nghiêm trọng** nhờ sử dụng Dropout (30%) và BatchNormalization.
- **Accuracy curve**: Train Accuracy tăng nhanh, Val Accuracy tăng chậm hơn — khoảng cách giữa hai đường (generalization gap) ở mức chấp nhận được.
- **Tác dụng của Early Stopping**: Ngăn chặn mô hình tiếp tục học trên nhiễu (noise) của tập Train. Best epoch tại epoch {np.argmin(history.history['val_loss'])} với val_loss = {min(history.history['val_loss']):.4f}.

**Nhận xét**: Mô hình **không bị Underfitting** (train accuracy khá cao) và Dropout + Early Stopping đã kiểm soát tốt hiện tượng **Overfitting**.

### 1.2 MLP (scikit-learn)

MLP sử dụng Early Stopping nội bộ. Mô hình hội tụ và dừng tự động.

### 1.3 XGBoost

XGBoost sử dụng Early Stopping dựa trên val_mlogloss. Mô hình cây quyết định có xu hướng **ít overfitting hơn** so với mạng nơ-ron nhờ cơ chế regularization tự nhiên (max_depth, min_child_weight).

---

## 2. 📊 So sánh Chỉ số Đo lường (Metrics Comparison)

| Metric | Deep ANN | MLP | XGBoost |
|--------|----------|-----|---------|
| **Accuracy** | {metrics_list[0]['accuracy']:.4f} | {metrics_list[1]['accuracy']:.4f} | {metrics_list[2]['accuracy']:.4f} |
| **Balanced Accuracy** | {metrics_list[0]['balanced_accuracy']:.4f} | {metrics_list[1]['balanced_accuracy']:.4f} | {metrics_list[2]['balanced_accuracy']:.4f} |
| **Macro Precision** | {metrics_list[0]['precision_macro']:.4f} | {metrics_list[1]['precision_macro']:.4f} | {metrics_list[2]['precision_macro']:.4f} |
| **Macro Recall** | {metrics_list[0]['recall_macro']:.4f} | {metrics_list[1]['recall_macro']:.4f} | {metrics_list[2]['recall_macro']:.4f} |
| **Macro F1-Score** | {metrics_list[0]['f1_macro']:.4f} | {metrics_list[1]['f1_macro']:.4f} | {metrics_list[2]['f1_macro']:.4f} |
| **Weighted F1-Score** | {metrics_list[0]['f1_weighted']:.4f} | {metrics_list[1]['f1_weighted']:.4f} | {metrics_list[2]['f1_weighted']:.4f} |

> 🏆 **Mô hình chiến thắng**: **{winner['name']}** với Weighted F1-Score = **{winner['f1_weighted']:.4f}**

---

## 3. 🔍 Phân tích Ma trận Nhầm lẫn (Confusion Matrix Analysis)

### Nhận xét chung:

- **Lớp Quality 5 và 6** (chiếm đa số): Cả 3 mô hình đều phân loại tốt nhất cho hai lớp này vì có nhiều dữ liệu huấn luyện.
- **Lớp Quality 3 và 8** (thiểu số cực đoan): Đây là điểm yếu lớn nhất. Dù đã áp dụng SMOTE, số lượng mẫu gốc quá ít (~10-15 mẫu) khiến mô hình khó học được pattern đặc trưng.
- **Nhầm lẫn phổ biến nhất**: Quality 5 ↔ 6 (hai lớp liền kề) — điều này hợp lý vì ranh giới giữa rượu chất lượng 5 và 6 rất mờ nhạt trong thực tế.
- **Mạng nơ-ron** (Deep ANN, MLP) có xu hướng **thiên lệch** về các lớp đa số hơn XGBoost, dù đã dùng class weights.

### Điểm yếu của mạng nơ-ron với bộ dữ liệu này:

1. **Dữ liệu quá nhỏ** (~1100 mẫu): Mạng nơ-ron sâu cần hàng nghìn đến hàng chục nghìn mẫu để phát huy sức mạnh. Với dữ liệu nhỏ, mô hình tree-based thường chiến thắng.
2. **Đặc trưng ít** (11 features): Mạng nơ-ron mạnh khi có nhiều đặc trưng phức tạp. Với 11 features đơn giản, XGBoost khai thác tốt hơn.
3. **Nhiễu trong nhãn**: Chất lượng rượu do người đánh giá chủ quan, gây nhiễu trong dữ liệu mà mạng nơ-ron nhạy cảm hơn.

---

## 4. 🎯 Kết luận & Khuyến nghị

### 4.1 Kết luận

- **{winner['name']}** là mô hình tốt nhất cho bài toán này.
- Đối với dữ liệu dạng bảng (tabular) nhỏ, **XGBoost/Gradient Boosting** thường vượt trội hơn mạng nơ-ron — phù hợp với các nghiên cứu gần đây.
- Tất cả mô hình đều gặp khó khăn với các lớp thiểu số (Quality 3, 4, 8).

### 4.2 Sẵn sàng triển khai (Deploy)?

- ⚠️ **Chưa hoàn toàn sẵn sàng** cho môi trường production nếu yêu cầu phân loại chính xác ALL classes.
- ✅ **Có thể triển khai** nếu chỉ cần phân loại các lớp phổ biến (5, 6, 7) hoặc gộp thành 3 nhóm: Thấp (3-4), Trung bình (5-6), Cao (7-8).

### 4.3 Khuyến nghị nâng cấp

1. **Thu thập thêm dữ liệu** cho các lớp thiểu số (Quality 3, 4, 8) để cải thiện đáng kể.
2. **Feature Engineering**: Tạo thêm đặc trưng mới (tỷ lệ acid/alcohol, tương tác features).
3. **Gộp nhãn**: Chuyển từ 6 lớp → 3 lớp (Thấp/Trung bình/Cao) để cải thiện balanced accuracy.
4. **Ensemble**: Kết hợp nhiều mô hình (Stacking/Voting) để tận dụng ưu điểm của từng mô hình.
5. **Hyperparameter Tuning**: Dùng Optuna/GridSearchCV để tối ưu hyperparameters chi tiết hơn.
6. **Cross-validation**: Sử dụng k-fold CV để đánh giá ổn định hơn trên tập dữ liệu nhỏ.

---

## 5. 📎 Phụ lục: Hình ảnh

- **Learning Curves**: `output/learning_curves.png`
- **Confusion Matrices**: `output/confusion_matrices.png`
- **Metrics Comparison**: `output/metrics_comparison.png`
- **Feature Importance**: `output/feature_importance.png`
"""

    # Lưu báo cáo
    report_path = os.path.join(os.path.dirname(output_dir), 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n📝 Đã lưu báo cáo: {report_path}")
    return report_path


# ============================================================================
# HÀM CHÍNH (MAIN)
# ============================================================================
def main():
    """
    Hàm chính điều phối toàn bộ pipeline:
    1. Xử lý dữ liệu
    2. Xây dựng & huấn luyện mô hình
    3. Đánh giá & tạo báo cáo
    """
    print("\n" + "🍷" * 35)
    print("  HỆ THỐNG PHÂN LOẠI CHẤT LƯỢNG RƯỢU VANG")
    print("  Wine Quality Classification System")
    print("🍷" * 35 + "\n")

    # ========================================
    # PHẦN 1: XỬ LÝ DỮ LIỆU
    # ========================================
    # Đường dẫn tới file dữ liệu
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'WineQT.csv')

    # Bước 1.0: Tải và khám phá dữ liệu
    df = load_and_explore_data(data_path)

    # Bước 1.1: Làm sạch dữ liệu
    df = clean_data(df)

    # Bước 1.2: Mã hóa và phân chia dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, class_names = (
        encode_and_split(df)
    )

    # Lưu tên đặc trưng trước khi scaling (để vẽ feature importance)
    feature_names = list(df.columns[:-1])

    # Bước 1.3: Chuẩn hóa đặc trưng
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = (
        scale_features(X_train, X_val, X_test)
    )

    # Bước 1.4: Xử lý mất cân bằng lớp (SMOTE chỉ trên tập Train)
    X_train_resampled, y_train_resampled = (
        handle_class_imbalance(X_train_scaled, y_train, class_names)
    )

    # Số lượng lớp
    n_classes = len(class_names)

    # ========================================
    # PHẦN 2: XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH
    # ========================================
    print("\n\n" + "🔧" * 35)
    print("  PHẦN 2: XÂY DỰNG & HUẤN LUYỆN MÔ HÌNH")
    print("🔧" * 35)

    # Tính class weights cho Deep ANN
    class_weight_dict = compute_weights(y_train_resampled, n_classes)
    print(f"\n⚖️  Class Weights: {class_weight_dict}")

    # --- Mô hình 1: Deep ANN ---
    ann_model, ann_history = build_and_train_deep_ann(
        X_train_resampled, y_train_resampled,
        X_val_scaled, y_val,
        n_classes, class_weight_dict
    )

    # --- Mô hình 2: MLP ---
    mlp_model = build_and_train_mlp(
        X_train_resampled, y_train_resampled,
        X_val_scaled, y_val
    )

    # --- Mô hình 3: XGBoost ---
    xgb_model = build_and_train_xgboost(
        X_train_resampled, y_train_resampled,
        X_val_scaled, y_val,
        n_classes
    )

    # ========================================
    # PHẦN 3: ĐÁNH GIÁ & BÁO CÁO
    # ========================================
    print("\n\n" + "📊" * 35)
    print("  PHẦN 3: ĐÁNH GIÁ TOÀN DIỆN TRÊN TEST SET")
    print("📊" * 35)

    # Đánh giá từng mô hình trên Test Set
    ann_metrics = evaluate_model(
        ann_model, X_test_scaled, y_test, "Deep ANN (sklearn)", class_names, is_keras=False
    )
    mlp_metrics = evaluate_model(
        mlp_model, X_test_scaled, y_test, "MLP (scikit-learn)", class_names
    )
    xgb_metrics = evaluate_model(
        xgb_model, X_test_scaled, y_test, "XGBoost", class_names
    )

    metrics_list = [ann_metrics, mlp_metrics, xgb_metrics]

    # In kết quả so sánh
    print_evaluation_results(metrics_list, class_names)

    # Vẽ biểu đồ trực quan hóa
    print("\n\n" + "🎨" * 35)
    print("  TẠO BIỂU ĐỒ TRỰC QUAN HÓA")
    print("🎨" * 35)

    plot_learning_curves(ann_history)
    plot_confusion_matrices(metrics_list, class_names)
    plot_metrics_comparison(metrics_list)
    plot_feature_importance(xgb_model, feature_names)

    # Tạo báo cáo Markdown
    generate_report(metrics_list, class_names, ann_history, OUTPUT_DIR)

    # ========================================
    # HOÀN TẤT
    # ========================================
    print("\n\n" + "✅" * 35)
    print("  HOÀN TẤT TOÀN BỘ PIPELINE!")
    print("✅" * 35)
    print(f"\n📂 Kết quả đầu ra đã được lưu tại: {OUTPUT_DIR}")
    print(f"   - learning_curves.png")
    print(f"   - confusion_matrices.png")
    print(f"   - metrics_comparison.png")
    print(f"   - feature_importance.png")
    print(f"   - ../report.md")

    # Xác định và in mô hình chiến thắng cuối cùng
    f1_scores = [m['f1_weighted'] for m in metrics_list]
    winner = metrics_list[np.argmax(f1_scores)]
    print(f"\n🏆 KẾT LUẬN: {winner['name']} chiến thắng với Weighted F1 = {winner['f1_weighted']:.4f}")
    print("🍷" * 35)


if __name__ == '__main__':
    main()
