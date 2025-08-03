import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Thêm các thư viện cần thiết cho giao diện vẽ
import tkinter as tk
from tkinter import Canvas, Frame, Button, Label
from PIL import Image, ImageDraw, ImageOps
# Thêm import cho các thành phần cần thiết để hiển thị ảnh trong tkinter
from PIL import ImageTk

# Xác định thiết bị tính toán
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Sử dụng thiết bị: {device}")

class ScaledUpDigitClassifierCNN(nn.Module):
    """
    Mạng CNN để phân loại chữ số viết tay MNIST
    """
    def __init__(self):
        super(ScaledUpDigitClassifierCNN, self).__init__()
        # Conv Layer 1: Nhận ảnh xám (1 channel), tạo 64 feature maps.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        # Conv Layer 2: 64 input -> 64 output feature maps.
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        # Conv Layer 3: 64 input -> 128 output feature maps.
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)

        # Conv Layer 4: 128 input -> 128 output feature maps.
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)

        # Dropout cho phần tích chập để tránh overfitting.
        self.dropout_conv = nn.Dropout2d(p=0.25)

        # Sau 2 lần MaxPool2d, kích thước ảnh là 7x7.
        # Tổng số feature: 128 * 7 * 7 = 6272
        self.fc1 = nn.Linear(in_features=128 * 7 * 7, out_features=512)
        self.bn_fc1 = nn.BatchNorm1d(num_features=512)
        self.dropout_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.bn_fc2 = nn.BatchNorm1d(num_features=256)
        self.dropout_fc2 = nn.Dropout(p=0.5)

        # Lớp output cuối cùng: 10 units cho 10 lớp chữ số (0-9).
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        """
        Định nghĩa cách dữ liệu đi qua mạng (forward pass).
        """
        # Conv1 -> BatchNorm -> ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        # Conv2 -> BatchNorm -> ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        # Max Pooling: Giảm kích thước không gian xuống 1 nửa (28x28 -> 14x14).
        x = F.max_pool2d(input=x, kernel_size=2)
        # Dropout để tránh overfitting.
        x = self.dropout_conv(x)

        # Conv3 -> BatchNorm -> ReLU
        x = F.relu(self.bn3(self.conv3(x)))
        # Conv4 -> BatchNorm -> ReLU
        x = F.relu(self.bn4(self.conv4(x)))
        # Max Pooling: (14x14 -> 7x7).
        x = F.max_pool2d(input=x, kernel_size=2)
        # Dropout.
        x = self.dropout_conv(x)

        # Flatten tensor từ 4D (batch, channel, height, width) về 2D (batch, features).
        x = torch.flatten(input=x, start_dim=1)

        # FC1 -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        # FC2 -> BatchNorm -> ReLU -> Dropout
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)

        # Lớp output cuối cùng. Trả về logits (raw scores).
        x = self.fc3(x)

        return x

class AnalysisNet(nn.Module):
    """
    Mô hình phụ trợ để lấy feature maps trung gian từ mô hình đã huấn luyện.
    """
    def __init__(self, original_model):
        super(AnalysisNet, self).__init__()
        # Sao chép tất cả các lớp từ mô hình gốc (ScaledUpDigitClassifierCNN)
        # Conv Block 1
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.conv2 = original_model.conv2
        self.bn2 = original_model.bn2

        # Conv Block 2
        self.conv3 = original_model.conv3
        self.bn3 = original_model.bn3
        self.conv4 = original_model.conv4
        self.bn4 = original_model.bn4

        # Các lớp khác
        self.dropout_conv = original_model.dropout_conv
        self.fc1 = original_model.fc1
        self.bn_fc1 = original_model.bn_fc1
        self.dropout_fc1 = original_model.dropout_fc1
        self.fc2 = original_model.fc2
        self.bn_fc2 = original_model.bn_fc2
        self.dropout_fc2 = original_model.dropout_fc2
        self.fc3 = original_model.fc3

    def forward(self, x):
        """
        Forward pass với việc lưu lại feature maps trung gian.
        """
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        conv1_out = x # Lưu feature maps sau block 1
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        # Conv Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        conv2_out = x # Lưu feature maps sau block 2
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout_conv(x)

        # Fully Connected Layers
        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        final_out = self.fc3(x) # Logits cuối cùng

        # Trả về output và các feature map trung gian
        return final_out, conv1_out, conv2_out

def load_trained_model(model_path="best_digit_classifier.pth"):
    """
    Tải mô hình đã được huấn luyện
    
    Args:
        model_path (str): Đường dẫn đến file mô hình
        
    Returns:
        model (ScaledUpDigitClassifierCNN): Mô hình đã được tải trọng số
    """
    print("[INFO] Đang tải mô hình tốt nhất đã lưu...")
    try:
        model = ScaledUpDigitClassifierCNN().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("[INFO] ✅ Đã tải xong mô hình tốt nhất.")
        return model
    except FileNotFoundError:
        print("[LỖI] Không tìm thấy file 'best_digit_classifier.pth'.")
        print("      Hãy đảm bảo rằng bạn đã huấn luyện mô hình và lưu lại trọng số.")
        return None
    except Exception as e:
        print(f"[LỖI] {e}")
        print("      Đang thử tải mà không dùng weights_only...")
        try:
            model = ScaledUpDigitClassifierCNN().to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("[INFO] ✅ Đã tải xong mô hình (không dùng weights_only).")
            return model
        except Exception as e2:
            print(f"[LỖI] Không thể tải mô hình: {e2}")
            return None

def predict_digit(model, image_array):
    """
    Dự đoán chữ số từ mảng hình ảnh 28x28
    
    Args:
        model (ScaledUpDigitClassifierCNN): Mô hình đã được huấn luyện
        image_array (np.ndarray): Mảng numpy 28x28 chứa hình ảnh
        
    Returns:
        predicted_class (int): Lớp được dự đoán (0-9)
        confidence (float): Độ tin cậy của dự đoán
        all_probabilities (np.ndarray): Xác suất cho tất cả 10 lớp
    """
    # Chuyển numpy array thành tensor PyTorch
    tensor_image = torch.from_numpy(image_array).float()
    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    normalized_tensor = (tensor_image - 0.1307) / 0.3081  # Normalize giống MNIST

    model.eval()
    with torch.no_grad():
        normalized_tensor = normalized_tensor.to(device)
        output = model(normalized_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()

        # TRẢ VỀ XÁC SUẤT CHO TẤT CẢ 10 LỚP
        all_probabilities = probabilities[0].cpu().numpy()

    return predicted_class, confidence, all_probabilities

def visualize_prediction(model, image_array, true_label=None):
    """
    Trực quan hóa kết quả dự đoán và feature maps
    
    Args:
        model (ScaledUpDigitClassifierCNN): Mô hình đã được huấn luyện
        image_array (np.ndarray): Mảng numpy 28x28 chứa hình ảnh
        true_label (int, optional): Nhãn thật (nếu có)
    """
    print("[INFO] Bắt đầu phân tích hình ảnh...")

    # --- Tạo mô hình phân tích ---
    analysis_model = AnalysisNet(model).to(device)
    analysis_model.eval()  # Đặt về chế độ đánh giá

    # Chuyển numpy array thành tensor PyTorch
    tensor_image = torch.from_numpy(image_array).float()
    tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    normalized_tensor = (tensor_image - 0.1307) / 0.3081  # Normalize giống MNIST

    with torch.no_grad():  # Tắt gradient
        normalized_tensor = normalized_tensor.to(device)
        # Forward pass và nhận output + feature maps
        logits, conv1_features, conv2_features = analysis_model(normalized_tensor)

        # --- 1. Tính toán dự đoán cuối cùng ---
        probabilities = F.softmax(logits, dim=1)  # Chuyển logits thành xác suất
        predicted_class = logits.argmax(dim=1).item()  # Lớp có xác suất cao nhất
        confidence = probabilities[0][predicted_class].item()  # Độ tin cậy
        all_probabilities = probabilities[0].cpu().numpy()  # Xác suất cho tất cả 10 chữ số

        # --- 2. Trực quan hóa kết quả ---
        # Tạo figure lớn để chứa nhiều hình ảnh
        fig = plt.figure(figsize=(20, 12))

        # a. Hình ảnh đầu vào (chiếm 2 cột)
        ax_input = plt.subplot2grid((4, 10), (0, 0), rowspan=2, colspan=2)
        img_to_show = normalized_tensor.cpu().squeeze()
        # "Unnormalize" để hiển thị hình ảnh tự nhiên hơn
        img_to_show = img_to_show * 0.3081 + 0.1307
        ax_input.imshow(img_to_show, cmap='gray')
        title = f'Dự đoán: {predicted_class}\n(Độ tin cậy: {confidence:.2f})'
        if true_label is not None:
            title += f'\nNhãn thật: {true_label}'
        ax_input.set_title(title, fontsize=12)
        ax_input.axis('off')

        # b. 16 Feature maps đầu tiên của Conv1
        conv1_cpu = conv1_features.cpu().squeeze()
        for i in range(min(16, conv1_cpu.shape[0])):  # Đảm bảo không vượt quá số lượng feature maps
            row = i // 4
            col = 2 + (i % 4)
            ax = plt.subplot2grid((4, 10), (row, col))
            ax.imshow(conv1_cpu[i], cmap='viridis')
            ax.set_title(f'Conv1_{i}', fontsize=9)
            ax.axis('off')

        # c. 16 Feature maps đầu tiên của Conv2
        conv2_cpu = conv2_features.cpu().squeeze()
        for i in range(min(16, conv2_cpu.shape[0])):  # Đảm bảo không vượt quá số lượng feature maps
            row = 2 + (i // 8)  # Hàng 2 hoặc 3
            col = i % 8         # Cột 0-7
            ax = plt.subplot2grid((4, 10), (row, col))
            ax.imshow(conv2_cpu[i], cmap='viridis')
            ax.set_title(f'Conv2_{i}', fontsize=9)
            ax.axis('off')

        plt.tight_layout(pad=1.0)
        plt.show()

        # --- 3. In kết quả phân tích bằng text ---
        print(f"\n[KẾT QUẢ DỰ ĐOÁN]")
        print(f"  - Dự đoán cuối cùng: Chữ số {predicted_class}")
        print(f"  - Độ tin cậy: {confidence:.4f}")

        # THÊM BẢNG XÁC SUẤT CHO TẤT CẢ 10 CHỮ SỐ
        print("\n[XÁC SUẤT CHO TỪNG CHỮ SỐ]")
        for i in range(10):
            prob = all_probabilities[i]
            print(f"  - Chữ số {i}: {prob:.4f} ({prob*100:.2f}%)")

        if true_label is not None:
            print(f"\n[NHÃN THẬT] {true_label}")
        print("-" * 40)

        plt.tight_layout()
        plt.show()

    return predicted_class, confidence

# Thêm lớp giao diện vẽ
class DrawingApp:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Vẽ Chữ Số - Nhận Dạng Chữ Số Viết Tay")
        
        # Kích thước canvas
        self.canvas_width = 280
        self.canvas_height = 280
        
        # Tạo khung chính
        self.main_frame = Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Tiêu đề
        title_label = Label(self.main_frame, text="Vẽ một chữ số (0-9)", font=("Arial", 16))
        title_label.pack(pady=10)
        
        # Tạo canvas để vẽ
        self.canvas = Canvas(
            self.main_frame, 
            width=self.canvas_width, 
            height=self.canvas_height, 
            bg="black",
            cursor="cross"
        )
        self.canvas.pack(pady=10)
        
        # Liên kết sự kiện chuột với canvas
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)
        
        # Tạo ảnh PIL và đối tượng vẽ để lưu vết vẽ
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        
        # Tạo các nút điều khiển
        button_frame = Frame(self.main_frame)
        button_frame.pack(pady=10)
        
        # Nút dự đoán
        self.predict_button = Button(
            button_frame, 
            text="Dự đoán", 
            command=self.predict_digit,
            font=("Arial", 12),
            bg="lightgreen",
            padx=20
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)
        
        # Nút xóa
        clear_button = Button(
            button_frame, 
            text="Xóa", 
            command=self.clear_canvas,
            font=("Arial", 12),
            bg="lightcoral",
            padx=20
        )
        clear_button.pack(side=tk.LEFT, padx=5)
        
        # Hiển thị kết quả
        self.result_label = Label(
            self.main_frame, 
            text="Kết quả sẽ hiển thị ở đây", 
            font=("Arial", 14),
            fg="blue"
        )
        self.result_label.pack(pady=10)
        
        # Khung để hiển thị ảnh đã xử lý và kết quả phân tích
        self.analysis_frame = Frame(self.main_frame)
        self.analysis_frame.pack(pady=10)
        
        # Label để hiển thị ảnh 28x28
        self.processed_image_label = Label(self.analysis_frame, text="Ảnh 28x28 sẽ hiển thị ở đây")
        self.processed_image_label.grid(row=0, column=0, padx=10)
        
        # Label để hiển thị kết quả phân tích
        self.analysis_result_label = Label(self.analysis_frame, text="Kết quả phân tích sẽ hiển thị ở đây", 
                                          font=("Arial", 10), justify=tk.LEFT)
        self.analysis_result_label.grid(row=0, column=1, padx=10)
        
        # Ma trận numpy để lưu hình ảnh 28x28
        self.drawing_array = np.zeros((28, 28), dtype=np.float32)
        
        # Vị trí chuột trước đó
        self.last_x, self.last_y = None, None

    def paint(self, event):
        """
        Xử lý sự kiện vẽ khi kéo chuột
        """
        x, y = event.x, event.y
        
        # Vẽ trên canvas tkinter
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, x, y,
                width=20, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE
            )
            
            # Vẽ trên ảnh PIL
            self.draw.line(
                [self.last_x, self.last_y, x, y],
                fill="white", width=20
            )
        
        self.last_x, self.last_y = x, y
        
        # Cập nhật mảng drawing_array
        self.update_drawing_array()

    def update_drawing_array(self):
        """
        Cập nhật mảng numpy 28x28 từ ảnh PIL
        """
        # Thay đổi kích thước ảnh về 28x28
        resized_image = self.image.resize((28, 28))
        
        # Chuyển đổi sang mảng numpy và chuẩn hóa
        self.drawing_array = np.array(resized_image, dtype=np.float32) / 255.0

    def clear_canvas(self):
        """
        Xóa toàn bộ canvas và đặt lại trạng thái
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.drawing_array = np.zeros((28, 28), dtype=np.float32)
        self.last_x, self.last_y = None, None
        self.result_label.config(text="Đã xóa. Vẽ một chữ số mới và nhấn 'Dự đoán'.")

    def predict_digit(self):
        """
        Dự đoán chữ số từ hình vẽ hiện tại và hiển thị kết quả phân tích
        """
        # Kiểm tra nếu canvas trống
        if np.all(self.drawing_array == 0):
            self.result_label.config(text="Vui lòng vẽ một chữ số trước khi dự đoán!")
            return
            
        # Tạo mô hình phân tích để lấy feature maps
        analysis_model = AnalysisNet(self.model).to(device)
        analysis_model.eval()
        
        # Chuyển numpy array thành tensor PyTorch
        tensor_image = torch.from_numpy(self.drawing_array).float()
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
        normalized_tensor = (tensor_image - 0.1307) / 0.3081  # Normalize giống MNIST

        with torch.no_grad():
            normalized_tensor = normalized_tensor.to(device)
            # Forward pass và nhận output + feature maps
            logits, conv1_features, conv2_features = analysis_model(normalized_tensor)

            # Tính toán dự đoán cuối cùng
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            all_probabilities = probabilities[0].cpu().numpy()

            # Hiển thị kết quả trên giao diện
            result_text = f"Chữ số dự đoán: {predicted_class}\nĐộ tin cậy: {confidence:.4f} ({confidence*100:.2f}%)"
            self.result_label.config(text=result_text)
            
            # Hiển thị ảnh 28x28 đã xử lý
            self.display_processed_image()
            
            # Hiển thị kết quả phân tích
            self.display_analysis_results(predicted_class, all_probabilities)

    def display_processed_image(self):
        """
        Hiển thị ảnh 28x28 đã xử lý trên giao diện
        """
        # Tạo ảnh PIL từ mảng numpy
        img_array = (self.drawing_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        
        # Phóng to ảnh lên để dễ xem (từ 28x28 lên 140x140)
        img = img.resize((140, 140), Image.NEAREST)
        
        # Chuyển đổi sang định dạng phù hợp với tkinter
        photo = ImageTk.PhotoImage(img)
        
        # Hiển thị ảnh
        self.processed_image_label.config(image=photo, text="")
        self.processed_image_label.image = photo  # Giữ tham chiếu để ảnh không bị xóa

    def display_analysis_results(self, predicted_class, all_probabilities):
        """
        Hiển thị kết quả phân tích chi tiết
        """
        # Tạo chuỗi kết quả chi tiết
        result_text = "XÁC SUẤT CHO TỪNG CHỮ SỐ:\n"
        for i in range(10):
            prob = all_probabilities[i]
            marker = ">>> " if i == predicted_class else "    "
            result_text += f"{marker}Chữ số {i}: {prob:.4f} ({prob*100:.2f}%)\n"
        
        # Hiển thị kết quả
        self.analysis_result_label.config(text=result_text)

    def run(self):
        """
        Chạy ứng dụng
        """
        self.root.mainloop()

def main():
    """
    Hàm chính để chạy ứng dụng
    """
    print("=== ỨNG DỤNG NHẬN DẠNG CHỮ SỐ VIẾT TAY ===")
    
    # Tải mô hình đã được huấn luyện
    model = load_trained_model()
    if model is None:
        print("[LỖI] Không thể tải mô hình. Chương trình sẽ kết thúc.")
        return
    
    # Khởi tạo và chạy ứng dụng vẽ
    print("\n[Khởi tạo ứng dụng vẽ]")
    app = DrawingApp(model)
    print("✅ Đã khởi tạo ứng dụng vẽ.")
    print("\nHướng dẫn:")
    print("1. Vẽ một chữ số (0-9) trên khung màu đen")
    print("2. Nhấn nút 'Dự đoán' để xem kết quả")
    print("3. Nhấn nút 'Xóa' để xóa và vẽ lại")
    
    # Chạy ứng dụng
    app.run()
    
    print("\n=== KẾT THÚC CHƯƠNG TRÌNH ===")

if __name__ == "__main__":
    main()