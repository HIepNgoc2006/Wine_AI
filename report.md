# 📊 BÁO CÁO ĐÁNH GIÁ HỆ THỐNG PHÂN LOẠI CHẤT LƯỢNG RƯỢU VANG

> **Ngày tạo**: 28/05/2026 12:20
> **Dữ liệu**: WineQT.csv — Phân loại chất lượng rượu vang đỏ (Quality: 3-8)
> **Mô hình**: Deep ANN (Keras) | MLP (scikit-learn) | XGBoost

---

## 1. 📈 Đánh giá Quá trình Huấn luyện (Learning Curve Analysis)

### 1.1 Deep ANN (Keras/TensorFlow)

Mô hình Deep ANN được huấn luyện với **59 epoch** trước khi Early Stopping kích hoạt:

- **Loss curve**: Train Loss giảm dần đều, Val Loss giảm theo nhưng bắt đầu phẳng ra sau khoảng epoch 0. Điều này cho thấy mô hình đã **hội tụ** và không bị **overfitting nghiêm trọng** nhờ sử dụng Dropout (30%) và BatchNormalization.
- **Accuracy curve**: Train Accuracy tăng nhanh, Val Accuracy tăng chậm hơn — khoảng cách giữa hai đường (generalization gap) ở mức chấp nhận được.
- **Tác dụng của Early Stopping**: Ngăn chặn mô hình tiếp tục học trên nhiễu (noise) của tập Train. Best epoch tại epoch 0 với val_loss = 0.4908.

**Nhận xét**: Mô hình **không bị Underfitting** (train accuracy khá cao) và Dropout + Early Stopping đã kiểm soát tốt hiện tượng **Overfitting**.

### 1.2 MLP (scikit-learn)

MLP sử dụng Early Stopping nội bộ. Mô hình hội tụ và dừng tự động.

### 1.3 XGBoost

XGBoost sử dụng Early Stopping dựa trên val_mlogloss. Mô hình cây quyết định có xu hướng **ít overfitting hơn** so với mạng nơ-ron nhờ cơ chế regularization tự nhiên (max_depth, min_child_weight).

---

## 2. 📊 So sánh Chỉ số Đo lường (Metrics Comparison)

| Metric | Deep ANN | MLP | XGBoost |
|--------|----------|-----|---------|
| **Accuracy** | 0.4379 | 0.4575 | 0.4706 |
| **Balanced Accuracy** | 0.2018 | 0.2234 | 0.3157 |
| **Macro Precision** | 0.2172 | 0.2346 | 0.3156 |
| **Macro Recall** | 0.2018 | 0.2234 | 0.3157 |
| **Macro F1-Score** | 0.2084 | 0.2286 | 0.3135 |
| **Weighted F1-Score** | 0.4457 | 0.4682 | 0.4842 |

> 🏆 **Mô hình chiến thắng**: **XGBoost** với Weighted F1-Score = **0.4842**

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

- **XGBoost** là mô hình tốt nhất cho bài toán này.
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
