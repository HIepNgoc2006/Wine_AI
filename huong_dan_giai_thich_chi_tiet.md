# TÀI LIỆU GIẢI THÍCH CHI TIẾT TOÀN BỘ DỰ ÁN WINE AI
*(Bí kíp đọc hiểu toàn bộ project - Dành cho các thành viên trong nhóm ôn tập trước khi báo cáo)*

---

## 1. MỤC ĐÍCH CỦA TÀI LIỆU
Tài liệu này được viết ra để **mọi thành viên trong nhóm** – kể cả người không trực tiếp code – cũng có thể nắm được trọn vẹn "linh hồn" của dự án. Khi báo cáo, thầy cô không quan tâm bạn thuộc lòng code ra sao, mà quan tâm bạn **hiểu tại sao lại làm như vậy**. Hãy đọc kỹ tài liệu này như một cuốn cẩm nang.

---

## 2. TỔNG QUAN DỰ ÁN (Project Overview)
- **Mục tiêu:** Dạy cho máy tính (AI) khả năng chấm điểm chất lượng rượu vang (từ điểm 3 đến điểm 8) dựa trên 11 thông số hóa học (như nồng độ cồn, độ pH, lượng đường, axit...).
- **Khó khăn lớn nhất:** Dữ liệu có tính chất **"Mất cân bằng cực đoan" (Extreme Imbalance)**. Nghĩa là số lượng chai vang đạt điểm 5 và 6 chiếm tới hơn 80% tổng số dữ liệu. Những chai quá dở (điểm 3) hoặc quá ngon (điểm 8) lại chỉ có đếm trên đầu ngón tay. Việc này khiến AI dễ bị "mù", chỉ chăm chăm đoán điểm 5 và 6.

---

## 3. GIẢI MÃ QUÁ TRÌNH XỬ LÝ DỮ LIỆU (Data Processing)
Quá trình xử lý dữ liệu quyết định đến 80% thành công của mô hình học máy. Nhóm đã áp dụng các chiến thuật sau:

### 3.1. Xử lý Missing Values (Dữ liệu bị thiếu) & Outliers (Giá trị ngoại lai)
- **Missing Values:** Nhóm dùng `Median` (Trung vị) thay cho `Mean` (Trung bình). **Lý do:** Trung bình rất dễ bị bóp méo bởi các giá trị ngoại lai, trong khi Trung vị thì an toàn và phản ánh đúng bản chất dữ liệu hơn.
- **Outliers:** Nếu xóa (Drop) các dòng có giá trị bất thường, nhóm sẽ mất đi lượng lớn dữ liệu (vốn dĩ đã rất ít, chỉ khoảng 1000 dòng). Nhóm dùng phương pháp **IQR (Interquartile Range)** kết hợp hàm `clip()`. Hàm `clip()` giống như một hàng rào: Những giá trị nào vượt quá hàng rào sẽ bị ép (giới hạn) về bằng mức hàng rào, giúp bảo vệ dữ liệu không bị xóa bỏ.

### 3.2. Chống "Rò rỉ dữ liệu" (Data Leakage) khi chia tập Data
Đây là điểm sáng giá nhất của bài toán. Nhóm chia dữ liệu thành 3 phần: Train (Học), Val (Kiểm định trong lúc học), và Test (Thi thật).
- **Stratified Split:** Nhóm dùng tham số `stratify` để đảm bảo tỷ lệ các nhãn (từ 3->8) được chia đều đặn vào 3 tập.
- **Data Leakage (Tuyệt đối quan trọng):** Nếu nhóm chuẩn hóa (StandardScaler) toàn bộ dữ liệu trước khi chia tập, mô hình sẽ biết trước thông tin của tập Test. Nhóm khắc phục bằng cách **CHỈ áp dụng lệnh `fit_transform` lên tập Train**. Tập Val và Test chỉ được `transform` dựa trên thông tin đã học từ Train. Tập Test hoàn toàn là dữ liệu "mù" với mô hình.

### 3.3. Xử lý mất cân bằng bằng SMOTE
SMOTE là thuật toán tự động sinh ra các dữ liệu nhân tạo cho các lớp bị thiếu (lớp 3, 4, 7, 8) dựa trên thuật toán K-láng giềng gần nhất. Nhờ đó, số lượng mẫu của tất cả các lớp trong tập Train đều được cân bằng (mỗi lớp 303 mẫu).
> **Lưu ý:** Tương tự như chuẩn hóa, SMOTE cũng CHỈ được gọi trên tập Train.

---

## 4. KIẾN TRÚC MÔ HÌNH VÀ CƠ CHẾ HUẤN LUYỆN (AI Architecture)
Nhóm xây dựng 3 mô hình để đối chứng, trọng tâm là mạng Nơ-ron sâu (Deep Neural Network) bằng TensorFlow/Keras.

### 4.1. Mạng Deep Neural Network (DNN)
Mạng được thiết kế theo hình phễu với 4 lớp ẩn (Hidden Layers): 256 -> 128 -> 64 -> 32 neurons.
Để mạng không bị **Overfitting (Học vẹt - thuộc lòng tập Train nhưng ra Test lại sai)**, nhóm tích hợp 3 cơ chế:
1. **Batch Normalization:** Chuẩn hóa luồng dữ liệu ngay giữa các layer, giúp quá trình lan truyền ngược (backpropagation) mượt mà hơn, đạo hàm không bị bùng nổ hay triệt tiêu.
2. **Dropout (0.3):** Ở mỗi vòng học, mô hình ngẫu nhiên "đánh ngất" (tắt đi) 30% số lượng nơ-ron. Điều này ép các nơ-ron còn lại phải làm việc chăm chỉ hơn, không được ỷ lại vào nhau, giúp mô hình mang tính khái quát cao.
3. **Early Stopping:** Cài đặt "Patience=20". Nghĩa là nếu sau 20 vòng học mà mô hình không tiến bộ thêm, nó sẽ tự động dừng lại để tiết kiệm thời gian và khôi phục về trạng thái tốt nhất. (Thực tế mô hình đã tự dừng ở vòng 41).

### 4.2. XGBoost - "Vua" của dữ liệu dạng bảng
Nhóm dùng XGBoost làm Baseline. Đây là mô hình dạng Cây Quyết định (Tree-based) kết hợp học tăng cường (Boosting). XGBoost thường vô địch trong các cuộc thi Kaggle liên quan đến dữ liệu dạng bảng (Tabular Data) vì nó tự động xử lý tốt các đặc trưng phi tuyến tính.

---

## 5. PHÂN TÍCH KẾT QUẢ THỰC NGHIỆM VÀ LỜI GIẢI THÍCH
### 5.1. Mô hình nào vô địch?
Nhóm KHÔNG sử dụng `Accuracy` làm thước đo chính vì Accuracy trên dữ liệu mất cân bằng là con số ảo tưởng (ví dụ mô hình cứ đoán tất cả là điểm 6 thì Accuracy vẫn cao). Nhóm sử dụng **Weighted F1-Score**.
Kết quả: XGBoost chiến thắng với F1-Score đạt **~50%**, cao hơn MLP và Deep ANN. Điều này chứng minh định lý: *"Với dữ liệu dạng bảng có cấu trúc nhỏ, Tree-based Models (XGBoost) làm việc hiệu quả hơn Neural Networks"*.

### 5.2. Tại sao độ chính xác cao nhất chỉ đạt 50%?
Giáo viên chắc chắn sẽ hỏi: *"Làm AI gì mà đúng có một nửa thế?"*. Hãy tự tin trả lời:
1. **Garbage in, Garbage out:** Việc đánh giá rượu vang là **cảm quan chủ quan của con người (chuyên gia nếm)**. Một chai rượu điểm 5 và điểm 6 có thành phần hóa học gần như Y HỆT NHAU. Vì thế, dữ liệu đầu vào vốn dĩ đã có sự chồng lấp (overlap) rất lớn, máy móc không thể tìm ra ranh giới rạch ròi.
2. **Thiếu dữ liệu gốc trầm trọng:** Dù đã dùng SMOTE để sinh dữ liệu nhân tạo cho lớp điểm 3 và 4, nhưng gốc rễ chỉ có chưa tới 10 mẫu. Việc nhân bản từ 10 mẫu lên 303 mẫu chỉ tạo ra các bản sao na ná nhau, chứ không đem lại tri thức mới. Khi ra tập Test thật, mô hình vẫn "chào thua" các chai điểm 3 và 4 (Precision = 0).
**=> Kết luận:** Con số 50% là kết quả **phản ánh trung thực tính chất cực khó của dữ liệu**, chứng tỏ nhóm không làm giả kết quả (không overfit).

### 5.3. Hướng giải quyết tương lai (Future Works)
Để làm dự án này thực tiễn hơn, nhóm đề xuất 2 hướng:
- **Hướng 1 (Binning):** Ép 6 nhãn (3,4,5,6,7,8) lại thành 3 nhãn: **Dở, Trung Bình, Tuyệt Hảo**. Thu hẹp nhãn sẽ triệt tiêu hoàn toàn sự mất cân bằng và độ chính xác có thể vọt lên 80-90%.
- **Hướng 2 (Regression):** Chuyển từ bài toán Phân loại (Classification) sang Hồi quy (Regression). Mô hình sẽ dự đoán ra số thực (ví dụ 5.6 điểm) rồi làm tròn thành 6. Cách này giúp mô hình hiểu được tính liên tục của điểm số.

---
**Chúc toàn team đọc kỹ và báo cáo thành công rực rỡ! Đừng học thuộc lòng, hãy đọc để "ngấm" logic nhé!**
