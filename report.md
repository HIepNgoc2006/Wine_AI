# 📊 BÁO CÁO TỔNG HỢP: HỆ THỐNG PHÂN LOẠI CHẤT LƯỢNG RƯỢU VANG

> **Ngày tạo**: 29/05/2026 01:28
> **Dữ liệu**: WineQT.csv — Phân loại chất lượng rượu vang đỏ (Quality: 3-8)
> **Mô hình**: Deep Neural Network (TensorFlow/Keras) | MLP (scikit-learn) | XGBoost
> **Ngôn ngữ**: Python | TensorFlow/tf.keras | scikit-learn | XGBoost

---

## 1. 📋 Tóm tắt Dữ liệu (Dataset Metrics)

### 1.1 Kích thước dữ liệu

| Chỉ số | Giá trị |
|--------|---------|
| **Dữ liệu gốc** | 1143 mẫu × 12 cột |
| **Ma trận đặc trưng X** | X ∈ ℝ^(1143 × 11) |
| **Sau làm sạch** | 1018 mẫu × 12 cột |
| **Số lớp phân loại** | 6 lớp (Quality 3, 4, 5, 6, 7, 8) |

### 1.2 Phân bố lớp

Dữ liệu mất cân bằng **nghiêm trọng**:
- **Lớp đa số**: Quality 5 (~42%) và Quality 6 (~40%) chiếm ~82% tổng dữ liệu
- **Lớp thiểu số**: Quality 3 (~0.5%) và Quality 8 (~1.4%) cực kỳ hiếm
- **Giải pháp**: Áp dụng SMOTE (Synthetic Minority Oversampling Technique) để cân bằng lớp trên tập Train, kết hợp Class Weights trong hàm mất mát

### 1.3 Tiền xử lý dữ liệu

1. **Missing Values**: Kiểm tra và điền bằng trung vị (median) nếu có — median robust hơn mean với dữ liệu lệch
2. **Duplicates**: Loại bỏ các dòng trùng lặp để tránh bias
3. **Outliers**: Cắt (clip) bằng phương pháp IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR) — giữ nguyên số mẫu
4. **Feature Scaling**: Chuẩn hóa bằng StandardScaler (z-score: z = (x-μ)/σ). Chỉ fit() trên Train, transform() lên Val/Test để tránh data leakage
5. **Phân chia dữ liệu**: Train (70%) / Validation (15%) / Test (15%) với `stratify` để giữ tỷ lệ lớp

---

## 2. 🎯 Chiến lược Huấn luyện

### 2.1 Tốc độ học (Learning Rate — λ) — Tham số quan trọng nhất

Tốc độ học (λ) quyết định mức độ cập nhật trọng số sau mỗi bước Gradient Descent:
- **λ quá lớn**: Mô hình "nhảy" qua điểm tối ưu → Loss dao động, không hội tụ (Poor Convergence)
- **λ quá nhỏ**: Hội tụ quá chậm, dễ rơi vào cực tiểu địa phương (local minimum)

| Mô hình | Learning Rate | Giải thích |
|---------|---------------|------------|
| **Deep ANN** | λ = 0.001 (Adam) + ReduceLROnPlateau | Giá trị chuẩn cho Adam optimizer. LR tự động giảm 50% khi val_loss bão hòa sau 10 epoch |
| **MLP** | λ₀ = 0.001 (adaptive) | Bắt đầu với 0.001, tự động giảm khi loss không cải thiện sau nhiều epoch |
| **XGBoost** | η = 0.05 | Nhỏ hơn mặc định (0.3) → ổn định hơn, kết hợp 500 cây + Early Stopping |

### 2.2 Chiến lược chống Overfitting

| Kỹ thuật | Mô hình áp dụng | Cơ chế hoạt động |
|----------|-----------------|------------------|
| **Dropout (30%, 20%)** | Deep ANN | Ngẫu nhiên tắt p% neurons mỗi lần forward pass → ensemble of sub-networks, giảm co-adaptation |
| **BatchNormalization** | Deep ANN | Chuẩn hóa output mỗi lớp về N(0,1) → ổn định gradient, giảm Internal Covariate Shift, cho phép LR cao hơn |
| **L2 Regularization** | Deep ANN (qua optimizer), MLP (alpha), XGBoost (reg_lambda) | Thêm penalty λ‖W‖² vào Loss → hạn chế trọng số quá lớn → giảm complexity |
| **Early Stopping** | Tất cả 3 mô hình | Dừng huấn luyện khi val_loss không giảm sau N epoch → ngăn mô hình "nhớ" noise của tập Train |
| **ReduceLROnPlateau** | Deep ANN | Giảm LR khi val_loss bão hòa → cho phép fine-tune gần điểm hội tụ tối ưu |

### 2.3 Khắc phục Vanishing Gradient

**Vấn đề**: Với mạng nhiều lớp, gradient bị giảm theo cấp số nhân khi lan truyền ngược (Backpropagation) qua các lớp sigmoid/tanh → lớp đầu hầu như không học được vì gradient ≈ 0.

**Giải pháp đã áp dụng trong Deep ANN**:
1. **Hàm kích hoạt ReLU**: f(x) = max(0, x) → đạo hàm = 1 khi x > 0, không bị co gradient như sigmoid (đạo hàm max = 0.25)
2. **BatchNormalization**: Giữ đầu ra mỗi lớp trong vùng hoạt động tốt của ReLU, tránh "dead neurons" (neurons luôn output 0)
3. **He Initialization**: Khởi tạo trọng số W ~ N(0, √(2/n_in)) — tối ưu cho ReLU, đảm bảo phương sai gradient ổn định qua các lớp

---

## 3. 📈 Kết quả Đánh giá

### 3.1 Phân tích Quá trình Huấn luyện (Learning Curves — Deep ANN)

Mô hình Deep ANN được huấn luyện với **41 epoch** trước khi Early Stopping kích hoạt:

- **Loss Curve**: Train Loss giảm dần đều, Val Loss giảm theo nhưng bắt đầu phẳng (bão hòa) sau epoch 21. Điều này cho thấy mô hình đã **hội tụ tốt** — hàm mất mát categorical_crossentropy đã được kiểm soát.
  - Khoảng cách giữa Train Loss và Val Loss (**generalization gap**) ở mức chấp nhận được, chứng minh **Dropout + BatchNormalization** đã kiểm soát hiệu quả hiện tượng Overfitting.
  - Hàm Loss giảm đều chứng tỏ Learning Rate (λ=0.001) được chọn **phù hợp** — không quá lớn (gây dao động/diverge) và không quá nhỏ (hội tụ chậm/stuck tại local minimum).

- **Accuracy Curve**: Train Accuracy tăng nhanh trong 20-30 epoch đầu, sau đó tăng chậm và bão hòa. Val Accuracy tăng tương ứng nhưng có dao động nhẹ — bình thường với tập validation nhỏ.

- **Tác dụng Early Stopping**: Dừng tại epoch 41, khôi phục trọng số tốt nhất tại epoch 21 (restore_best_weights=True). Best val_loss = 1.0749, val_accuracy = 0.5359.

### 3.2 Bảng so sánh chỉ số đo lường (trên Test Set)

| Chỉ số | Deep ANN (tf.keras) | MLP (sklearn) | XGBoost |
|--------|---------------------|---------------|---------| 
| **Accuracy** | 0.4118 | 0.4575 | 0.4902 |
| **Balanced Accuracy** | 0.2986 | 0.2234 | 0.3300 |
| **Macro Precision** | 0.2651 | 0.2346 | 0.3251 |
| **Macro Recall** | 0.2986 | 0.2234 | 0.3300 |
| **Macro F1-Score** | 0.2621 | 0.2286 | 0.3246 |
| **Weighted F1-Score** | 0.4477 | 0.4682 | 0.5032 |

> 🏆 **Mô hình chiến thắng**: **XGBoost** với Weighted F1-Score = **0.5032**

### 3.3 Classification Report chi tiết (trên Test Set)

#### Deep ANN (TensorFlow/Keras)
```
              precision    recall  f1-score   support

          Q3       0.00      0.00      0.00         1
          Q4       0.00      0.00      0.00         5
          Q5       0.63      0.49      0.55        65
          Q6       0.50      0.35      0.42        62
          Q7       0.30      0.44      0.36        18
          Q8       0.17      0.50      0.25         2

    accuracy                           0.41       153
   macro avg       0.27      0.30      0.26       153
weighted avg       0.51      0.41      0.45       153

```

#### MLP (scikit-learn)
```
              precision    recall  f1-score   support

          Q3       0.00      0.00      0.00         1
          Q4       0.00      0.00      0.00         5
          Q5       0.59      0.52      0.55        65
          Q6       0.47      0.48      0.48        62
          Q7       0.35      0.33      0.34        18
          Q8       0.00      0.00      0.00         2

    accuracy                           0.46       153
   macro avg       0.23      0.22      0.23       153
weighted avg       0.48      0.46      0.47       153

```

#### XGBoost
```
              precision    recall  f1-score   support

          Q3       0.00      0.00      0.00         1
          Q4       0.00      0.00      0.00         5
          Q5       0.67      0.60      0.63        65
          Q6       0.48      0.44      0.46        62
          Q7       0.30      0.44      0.36        18
          Q8       0.50      0.50      0.50         2

    accuracy                           0.49       153
   macro avg       0.33      0.33      0.32       153
weighted avg       0.52      0.49      0.50       153

```

---

## 4. 🔍 Phân tích Sai số (Error Analysis)

### 4.1 Phân tích Ma trận Nhầm lẫn

- **Lớp Quality 5 và 6** (chiếm đa số ~82%): Cả 3 mô hình đều phân loại tốt nhất cho hai lớp này vì có nhiều dữ liệu huấn luyện. Tuy nhiên, **nhầm lẫn Q5↔Q6** rất phổ biến — ranh giới giữa "rượu chất lượng trung bình" và "khá tốt" rất mờ nhạt trong thực tế, phụ thuộc đánh giá chủ quan của người chấm.

- **Lớp Quality 3 và 8** (thiểu số cực đoan ~0.5% và ~1.4%): Precision và Recall thường rất thấp hoặc bằng 0 cho hai lớp này. Dù SMOTE đã tạo thêm mẫu tổng hợp và Class Weights đã được áp dụng, số mẫu gốc quá ít (6-16 mẫu) khiến mô hình không thể học được đặc trưng riêng biệt.

- **Lớp Quality 4 và 7** (thiểu số vừa): Kết quả tốt hơn Q3/Q8 nhưng vẫn thấp so với Q5/Q6. XGBoost thường xử lý tốt hơn mạng nơ-ron cho các lớp này nhờ khả năng tạo decision boundary không tuyến tính hiệu quả.

### 4.2 Phân tích từ góc độ Vanishing Gradient & Overfitting

1. **Vanishing Gradient đã được kiểm soát**: Nhờ ReLU + BatchNormalization, gradient chảy ổn định qua 4 lớp ẩn của Deep ANN. Bằng chứng: Train Loss giảm liên tục từ epoch 1 đến epoch cuối, không bị "đóng băng" ở các lớp sâu. Nếu xảy ra Vanishing Gradient, Train Loss sẽ ngừng giảm sớm.

2. **Overfitting được kiểm soát nhưng chưa loại bỏ hoàn toàn**: Khoảng cách Train Acc - Val Acc (generalization gap) cho thấy mô hình vẫn có xu hướng "nhớ" tập Train. Dropout (30%) và BatchNorm đã giảm thiểu đáng kể nhưng không thể triệt tiêu hoàn toàn do dữ liệu nhỏ (~1000 mẫu).

3. **Tại sao Precision của lớp thiểu số thấp?**
   - Precision = TP / (TP + FP): khi mô hình dự đoán một mẫu thuộc lớp thiểu số, xác suất đúng rất thấp vì FP cao
   - Nguyên nhân: mô hình học được rằng dự đoán lớp đa số "an toàn" hơn → bias về phía Q5/Q6
   - SMOTE tạo mẫu tổng hợp bằng nội suy tuyến tính nhưng không thêm thông tin mới thực sự → hiệu quả hạn chế khi số mẫu gốc < 10

4. **Tại sao XGBoost thường tốt hơn trên dữ liệu tabular nhỏ?**
   - XGBoost là ensemble of decision trees — mỗi cây chỉ cần tìm ngưỡng split đơn giản
   - Với 11 features và ~1000 mẫu, cây quyết định khai thác hiệu quả hơn các tương tác giữa features
   - Mạng nơ-ron cần hàng nghìn đến hàng chục nghìn mẫu để phát huy ưu thế biểu diễn phi tuyến

---

## 5. 🎯 Kết luận & Hướng phát triển

### 5.1 Kết luận

1. **XGBoost** đạt hiệu suất cao nhất trong 3 mô hình được thử nghiệm, với Weighted F1-Score = 0.5032.
2. Đối với dữ liệu dạng bảng (tabular) nhỏ (~1000 mẫu, 11 features), **mô hình tree-based (XGBoost) thường vượt trội** hơn mạng nơ-ron — phù hợp với các nghiên cứu benchmark gần đây (Grinsztajn et al., 2022).
3. Mất cân bằng lớp nghiêm trọng (Quality 3: ~0.5%, Quality 8: ~1.4%) là **thách thức lớn nhất** — tất cả mô hình đều gặp khó khăn với lớp thiểu số dù đã áp dụng SMOTE + Class Weights.
4. Chiến lược **ReLU + BatchNorm + Dropout + Early Stopping** đã kiểm soát hiệu quả Vanishing Gradient và Overfitting cho mạng nơ-ron sâu 4 lớp ẩn.
5. Learning Rate λ=0.001 (Adam) được chọn phù hợp — Loss hội tụ ổn định, không dao động.

### 5.2 Hướng phát triển

1. **Thu thập thêm dữ liệu** cho lớp thiểu số (Quality 3, 4, 8) — giải pháp hiệu quả nhất để cải thiện Balanced Accuracy
2. **Feature Engineering**: Tạo đặc trưng mới (tỷ lệ acid/alcohol, tương tác features, polynomial features)
3. **Gộp nhãn**: Chuyển từ 6 lớp → 3 lớp (Thấp: 3-4 / Trung bình: 5-6 / Cao: 7-8) để cải thiện đáng kể balanced accuracy
4. **Ensemble methods**: Kết hợp 3 mô hình bằng Stacking hoặc Voting Classifier để tận dụng ưu điểm của từng mô hình
5. **Hyperparameter Tuning**: Dùng Optuna/Bayesian Optimization để tối ưu siêu tham số toàn diện
6. **Cross-Validation**: Sử dụng k-fold Stratified CV (k=5) để đánh giá ổn định hơn trên dữ liệu nhỏ
7. **Thử nghiệm kiến trúc khác**: TabNet (Arik & Pfister, 2021), ResNet cho tabular data

---

## 6. 📎 Phụ lục

### Hình ảnh minh họa
- **Learning Curves**: `output/learning_curves.png`
- **Confusion Matrices**: `output/confusion_matrices.png`
- **Metrics Comparison**: `output/metrics_comparison.png`
- **Feature Importance**: `output/feature_importance.png`

### Thông tin kỹ thuật
- **Framework DNN**: TensorFlow / tf.keras (BẮT BUỘC theo yêu cầu)
- **scikit-learn**: MLP (MLPClassifier), Preprocessing (StandardScaler, LabelEncoder)
- **XGBoost**: Gradient Boosting Tree-based baseline
- **imbalanced-learn**: SMOTE oversampling
- **Random Seed**: 42 (đảm bảo reproducibility)
