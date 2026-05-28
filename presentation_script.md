# KẾ HOẠCH TOÀN DIỆN: THUYẾT TRÌNH HỆ THỐNG PHÂN LOẠI RƯỢU VANG (WINE QUALITY CLASSIFICATION)

*Tài liệu này bao gồm TẤT CẢ mọi thứ bạn cần: (1) Phân công 5 người, (2) Thiết kế Slide, (3) Lời thoại, và (4) Giải thích code.*

---

## I. TỔNG QUAN PHÂN CÔNG (5 THÀNH VIÊN)

| Thành viên | Trọng tâm Thuyết trình | Slides đảm nhận |
| :--- | :--- | :--- |
| **Thành viên 1** | Giới thiệu bài toán & Chiến lược xử lý nhiễu dữ liệu. | Slide 1, 2, 3 |
| **Thành viên 2** | SMOTE & Tư duy chống Rò rỉ dữ liệu (Data Leakage). | Slide 4, 5 |
| **Thành viên 3** | Giải mã kiến trúc mạng Deep Neural Network (DNN). | Slide 6, 7 |
| **Thành viên 4** | Mô hình Baseline (MLP, XGBoost) & Phân tích Kết quả. | Slide 8, 9 |
| **Thành viên 5** | Phản biện kết quả (50%) & Đề xuất Hướng tương lai. | Slide 10, 11, 12 |

---

## II. HƯỚNG DẪN THIẾT KẾ SLIDE (DESIGN GUIDELINES)
*   **Màu sắc chủ đạo:** Tông màu trầm (Đỏ Bordeaux / Đen / Trắng) mang cảm giác cao cấp của rượu vang.
*   **Nguyên tắc:** Slide KHÔNG DÙNG NHIỀU CHỮ. Chỉ dùng Bullet points (các gạch đầu dòng ngắn), Biểu đồ (lấy từ thư mục `output`), và các đoạn Code ngắn gọn để minh họa.

---

## III. KỊCH BẢN CHI TIẾT TỪNG THÀNH VIÊN

### 👤 THÀNH VIÊN 1: Người mở màn & Xử lý Dữ liệu thô

**[Slide 1: Tiêu đề]**
*   **Thiết kế Slide:** Tên đề tài, Danh sách 5 thành viên. Nền mờ ly rượu vang đỏ.
*   **🗣️ Lời thoại:** *"Dạ em chào thầy/cô và các bạn. Hôm nay nhóm xin trình bày đề tài: Phân loại chất lượng rượu vang bằng AI. Em là [Tên], xin phép mở đầu báo cáo."*

**[Slide 2: Bài toán & Đặc trưng Dữ liệu (WineQT)]**
*   **Thiết kế Slide:** Bảng tóm tắt 11 đặc trưng (pH, cồn...). Một biểu đồ cột hiển thị sự chênh lệch nhãn (Mất cân bằng dữ liệu >80% ở lớp 5, 6).
*   **🗣️ Lời thoại:** *"Bài toán của nhóm là dự đoán điểm rượu vang (3-8) dựa trên 11 thông số hóa học. Thử thách lớn nhất là tính **mất cân bằng cực đoan**. Hơn 80% dữ liệu nằm ở điểm 5 và 6, trong khi điểm 3 và 8 lại quá khan hiếm."*

**[Slide 3: Tiền xử lý & Bảo tồn dữ liệu]**
*   **Thiết kế Slide:** Sơ đồ: Missing -> Median. Outliers -> IQR Clip (không xóa).
*   **🗣️ Lời thoại:** *"Với bộ data chỉ ~1000 mẫu, nguyên tắc số 1 của nhóm là 'Bảo tồn dữ liệu'. Nhóm dùng Median thay cho Mean để tránh nhiễu. Đặc biệt, nhóm không XÓA các giá trị ngoại lai mà dùng hàm Clip chặn biên."*
*   **💻 Code giải thích (Có thể cắt ảnh đưa lên slide):**
```python
# LỜI GIẢI THÍCH: Dùng clip() giới hạn [lower, upper] thay vì drop() 
# để không làm mất bất kỳ một mẫu dữ liệu quý giá nào.
df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

---

### 👤 THÀNH VIÊN 2: Chuyên gia Tiền xử lý & SMOTE

**[Slide 4: Chiến lược chia tập dữ liệu]**
*   **Thiết kế Slide:** Sơ đồ 3 khối: Train (70%), Val (15%), Test (15%). Nổi bật chữ `Stratify`.
*   **🗣️ Lời thoại:** *"Chào thầy cô, em là [Tên]. Để chia data, nhóm dùng kỹ thuật Stratified Split nhằm đảm bảo tỷ lệ các nhãn được phân bố đồng đều giữa các tập Train, Val và Test."*

**[Slide 5: SMOTE & Chống Rò rỉ dữ liệu (Cực kỳ quan trọng)]**
*   **Thiết kế Slide:** Biểu tượng CẢNH BÁO Data Leakage. Sơ đồ: Chỉ áp dụng SMOTE + StandardScaler lên tập TRAIN.
*   **🗣️ Lời thoại:** *"Để khắc phục mất cân bằng, nhóm dùng thuật toán SMOTE để sinh dữ liệu giả cho lớp thiểu số. Nhưng điểm cốt lõi nhất là: Nhóm **chỉ áp dụng StandardScaler và SMOTE trên tập Train**. Tập Val/Test được giữ nguyên để giả lập dữ liệu tương lai. Nếu làm ngược lại, mô hình sẽ bị ảo tưởng (Data Leakage)."*
*   **💻 Code giải thích (Có thể cắt ảnh đưa lên slide):**
```python
# LỜI GIẢI THÍCH: TUYỆT ĐỐI CHỈ dùng fit_transform() trên tập Train.
# Với Val và Test, chỉ dùng transform() (chỉ dựa vào độ lệch của Train)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# SMOTE cũng chỉ áp dụng trên Train
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

---

### 👤 THÀNH VIÊN 3: Chuyên gia Deep Learning

**[Slide 6: Kiến trúc Mạng Nơ-ron (DNN)]**
*   **Thiết kế Slide:** Sơ đồ khối mạng Neural: Input (11) -> 256 -> 128 -> 64 -> 32 -> Output (6).
*   **🗣️ Lời thoại:** *"Phần xây dựng mô hình, em là [Tên] xin phép trình bày về mạng Deep ANN bằng TensorFlow. Nhóm thiết kế 4 khối Hidden Layers thu nhỏ dần để chắt lọc đặc trưng hóa học của rượu."*

**[Slide 7: Ba Vũ khí chống Overfitting]**
*   **Thiết kế Slide:** 3 Icon: Batch Normalization, Dropout (30%), Early Stopping.
*   **🗣️ Lời thoại:** *"Để mạng không học vẹt, nhóm cài đặt 3 công cụ: (1) BatchNormalization giúp ổn định luồng đạo hàm, (2) Dropout tắt ngẫu nhiên 30% nơ-ron để AI bớt ỷ lại, và (3) Early Stopping tự động dừng nếu mô hình không còn cải thiện."*
*   **💻 Code giải thích (Có thể cắt ảnh đưa lên slide):**
```python
# LỜI GIẢI THÍCH: 1 block cơ bản của nhóm chứa đủ vũ khí chống Overfit
model.add(layers.Dense(256, kernel_initializer='he_normal'))
model.add(layers.BatchNormalization()) # Ổn định hội tụ
model.add(layers.Activation('relu'))   # Tránh triệt tiêu đạo hàm
model.add(layers.Dropout(0.3))         # Tắt ngẫu nhiên 30% nơ-ron
```

---

### 👤 THÀNH VIÊN 4: Mô hình Baseline & Đánh giá

**[Slide 8: Các mô hình Đối chứng]**
*   **Thiết kế Slide:** Logo Scikit-Learn (MLP) và XGBoost.
*   **🗣️ Lời thoại:** *"Em là [Tên]. Để so sánh khách quan, nhóm huấn luyện thêm 2 mô hình là MLP và XGBoost. XGBoost là mô hình dạng Cây (Tree-based) được mệnh danh là 'vua' của dữ liệu dạng bảng."*

**[Slide 9: Bảng xếp hạng Kết quả]**
*   **Thiết kế Slide:** Chèn biểu đồ `metrics_comparison.png` từ thư mục output. Đánh dấu đỏ cột Weighted F1-Score.
*   **🗣️ Lời thoại:** *"Với dữ liệu mất cân bằng, Accuracy (Độ chính xác) sẽ phản ánh sai sự thật. Nhóm quyết định dùng thước đo **Weighted F1-Score**. Kết quả, XGBoost giành chiến thắng tuyệt đối với F1 đạt hơn 50%, vượt qua Deep ANN."*
*   **💻 Code giải thích (Có thể cắt ảnh đưa lên slide):**
```python
# LỜI GIẢI THÍCH: Tính Weighted F1 (Kết hợp cả Precision và Recall, có trọng số)
metrics['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
```

---

### 👤 THÀNH VIÊN 5: Người phản biện & Hướng tương lai

**[Slide 10: Tại sao kết quả chỉ đạt 50%?]**
*   **Thiết kế Slide:** Biểu tượng con người (cảm quan nếm thử) vs Máy móc (hóa học). Chữ TO: "Garbage in, Garbage out".
*   **🗣️ Lời thoại:** *"Chào thầy cô, em là [Tên]. Chắc hẳn mọi người sẽ thắc mắc: 'Mô hình vô địch mà chỉ đạt 50%?'. Nhóm xin khẳng định 50% này là con số TRUNG THỰC. Việc chấm điểm rượu là do chuyên gia nếm (chủ quan). Các thành phần hóa học của chai điểm 5, 6, 7 chồng lấp lên nhau quá nhiều, khiến máy móc không thể tìm ra quy luật toán học tuyệt đối."*

**[Slide 11: Định hướng phát triển]**
*   **Thiết kế Slide:** 
    * Cách 1: Gộp nhãn (Binning): Kém - Trung bình - Ngon.
    * Cách 2: Hồi quy (Regression).
*   **🗣️ Lời thoại:** *"Để AI hoạt động thực tiễn hơn, nhóm đề xuất (1) Gộp 6 nhãn chất lượng lại thành 3 cấp độ: Kém, Trung bình, Ngon. (2) Đổi bài toán Phân loại thành Hồi quy (dự đoán điểm liên tục rồi làm tròn). Hai cách này sẽ dễ dàng đẩy độ chính xác lên trên 80%."*

**[Slide 12: Q&A]**
*   **Thiết kế Slide:** Chữ "THANK YOU".
*   **🗣️ Lời thoại:** *"Bài thuyết trình của nhóm xin khép lại. Cảm ơn thầy cô đã lắng nghe. Chúng em rất mong nhận được câu hỏi phản biện ạ."*

---
## 💡 LƯU Ý QUAN TRỌNG CHO CẢ NHÓM:
1. File này đã bao gồm mọi thứ. Nhóm trưởng có thể copy các đoạn chữ trong phần **Thiết kế Slide** quăng lên Canva hoặc PowerPoint.
2. Các đoạn **Code giải thích**, hãy chụp màn hình code (bằng công cụ như Snipping Tool hoặc Carbon.now.sh cho đẹp) rồi chèn vào slide. Nhớ giải thích đúng như **Lời thoại** đã ghi.
