import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                           QComboBox, QSpinBox, QProgressBar, QMessageBox,
                           QLineEdit, QTextEdit, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
import yaml
import torch
import cv2

# 支持的图片格式
SUPPORTED_FORMATS = IMG_FORMATS

class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model_path, data_yaml, epochs, imgsz):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz

    def run(self):
        try:
            model = YOLO(self.model_path)
            
            # 训练模型
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                plots=True,  # 生成训练过程图表
                save=True,   # 保存训练结果
                device='0' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
            )
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class ImageLabel(QLabel):
    """图片标注控件"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.boxes = []  # 存储标注框
        self.current_box = None
        self.drawing = False
        self.current_class = 0
        self.image_path = None
        self.scale = 1.0  # 缩放比例
        self.offset = QPoint(0, 0)  # 图片偏移量
        self.dragging = False  # 是否正在拖动图片
        self.drag_start = None  # 拖动起始点
        self.original_pixmap = None  # 原始图片
        self.pixmap = None  # 缩放后的图片
        self.setMouseTracking(True)
        self.setStyleSheet("border: 1px solid black")
        self.setAlignment(Qt.AlignCenter)

    def setImage(self, image_path):
        """设置要显示的图片"""
        self.image_path = image_path
        self.original_pixmap = QPixmap(image_path)
        if self.original_pixmap.isNull():
            print(f"Failed to load image: {image_path}")
            return
        self.scale = 1.0
        self.offset = QPoint(0, 0)  # 重置偏移量
        self.updatePixmap()
        self.boxes = []

    def updatePixmap(self):
        """更新显示的图片"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
            
        # 计算缩放后的尺寸
        scaled_width = int(self.original_pixmap.width() * self.scale)
        scaled_height = int(self.original_pixmap.height() * self.scale)
        
        # 缩放图片
        self.pixmap = self.original_pixmap.scaled(scaled_width, scaled_height, 
                                                Qt.KeepAspectRatio, 
                                                Qt.SmoothTransformation)
        
        # 更新所有标注框的位置
        for box in self.boxes:
            # 获取相对坐标
            x_center, y_center, width, height = box['box']
            
            # 转换为像素坐标
            x = int((x_center - width/2) * scaled_width)
            y = int((y_center - height/2) * scaled_height)
            w = int(width * scaled_width)
            h = int(height * scaled_height)
            
            # 更新矩形
            box['rect'] = QRect(x, y, w, h)
            
        self.update()  # 不直接设置pixmap，而是通过paintEvent绘制

    def paintEvent(self, event):
        """绘制事件"""
        super().paintEvent(event)
        
        if self.pixmap is None or self.pixmap.isNull():
            return

        painter = QPainter(self)
        
        # 计算图片在控件中的居中位置
        pixmap_width = self.pixmap.width()
        pixmap_height = self.pixmap.height()
        x = (self.width() - pixmap_width) // 2
        y = (self.height() - pixmap_height) // 2
        
        # 绘制图片（应用偏移量）
        painter.drawPixmap(x + self.offset.x(), y + self.offset.y(), self.pixmap)
        
        # 设置画笔
        painter.setPen(QPen(Qt.red, max(1, int(2 * self.scale))))

        # 绘制已保存的框和类别名称
        main_window = self.window()
        if not hasattr(main_window, 'class_names') or not main_window.class_names:
            return
            
        for box in self.boxes:
            # 调整矩形位置以匹配图片位置
            adjusted_rect = QRect(
                box['rect'].x() + x + self.offset.x(),
                box['rect'].y() + y + self.offset.y(),
                box['rect'].width(),
                box['rect'].height()
            )
            painter.drawRect(adjusted_rect)
            try:
                class_name = main_window.class_names[box['class']]
                painter.drawText(adjusted_rect.topLeft() + QPoint(5, -5), class_name)
            except IndexError:
                continue

        # 绘制正在画的框
        if self.drawing and self.current_box:
            adjusted_rect = QRect(
                self.current_box.x() + x + self.offset.x(),
                self.current_box.y() + y + self.offset.y(),
                self.current_box.width(),
                self.current_box.height()
            )
            painter.drawRect(adjusted_rect)

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if self.pixmap is None or self.pixmap.isNull():
            return
            
        # 计算图片在控件中的居中位置
        x = (self.width() - self.pixmap.width()) // 2
        y = (self.height() - self.pixmap.height()) // 2
            
        if event.button() == Qt.LeftButton:
            # 开始绘制标注框
            self.drawing = True
            # 将鼠标位置转换为相对于图片的坐标
            self.start_point = QPoint(
                event.pos().x() - x - self.offset.x(),
                event.pos().y() - y - self.offset.y()
            )
            self.current_box = QRect(self.start_point, self.start_point)
        elif event.button() == Qt.RightButton:
            # 开始拖动图片
            self.dragging = True
            self.drag_start = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.pixmap is None or self.pixmap.isNull():
            return
            
        # 计算图片在控件中的居中位置
        x = (self.width() - self.pixmap.width()) // 2
        y = (self.height() - self.pixmap.height()) // 2
            
        if self.drawing:
            # 更新标注框，使用相对于图片的坐标
            current_pos = QPoint(
                event.pos().x() - x - self.offset.x(),
                event.pos().y() - y - self.offset.y()
            )
            self.current_box = QRect(self.start_point, current_pos).normalized()
            self.update()
        elif self.dragging:
            # 更新图片位置
            delta = event.pos() - self.drag_start
            self.offset += delta
            self.drag_start = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            if self.current_box and self.current_box.width() > 10 and self.current_box.height() > 10:
                # 转换为相对坐标
                pixmap_width = self.pixmap.width()
                pixmap_height = self.pixmap.height()
                
                # 计算相对坐标（考虑偏移量���
                box_x = self.current_box.x() / pixmap_width
                box_y = self.current_box.y() / pixmap_height
                box_width = self.current_box.width() / pixmap_width
                box_height = self.current_box.height() / pixmap_height
                
                # 添加标注框，保存相对坐标和当前像素坐标
                self.boxes.append({
                    'class': self.current_class,
                    'box': [box_x + box_width/2, box_y + box_height/2, box_width, box_height],
                    'rect': QRect(self.current_box)  # 保存当前像素坐标的矩形
                })
            
            self.current_box = None
            self.update()
        elif event.button() == Qt.RightButton:
            self.dragging = False

    def wheelEvent(self, event):
        """鼠标滚轮事件 - 用于缩放"""
        if self.original_pixmap is None or self.original_pixmap.isNull():
            return
            
        # 获取鼠标位置
        pos = event.pos()
        
        # 计算缩放比例
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1  # 放大10%
        else:
            self.scale *= 0.9  # 缩小10%
        
        # 限制缩放范围
        self.scale = max(0.1, min(5.0, self.scale))
        
        self.updatePixmap()

    def save_labels(self, label_path):
        """保存标注到文件"""
        with open(label_path, 'w', encoding='utf-8') as f:
            for box in self.boxes:
                f.write(f"{box['class']} {box['box'][0]} {box['box'][1]} {box['box'][2]} {box['box'][3]}\n")

    def load_labels(self, label_path):
        """加载已有标注"""
        if not os.path.exists(label_path) or not hasattr(self, 'pixmap'):
            return
        
        self.boxes = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                
                # 转换回像素坐标
                pixmap_width = self.pixmap.width()
                pixmap_height = self.pixmap.height()
                
                x = int((x_center - width/2) * pixmap_width)
                y = int((y_center - height/2) * pixmap_height)
                w = int(width * pixmap_width)
                h = int(height * pixmap_height)
                
                # 保存标注框，同时保存相对坐标和像素坐标
                self.boxes.append({
                    'class': int(class_id),
                    'box': [x_center, y_center, width, height],  # 保��相对坐标
                    'rect': QRect(x, y, w, h)  # 保存像素坐标
                })
        self.update()

    def clear_boxes(self):
        """清除所有标注框"""
        self.boxes = []
        self.update()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 训练器")
        self.setMinimumSize(1200, 800)
        
        # 存储类别名称
        self.class_names = []
        
        # 创建主口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # 左侧制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        layout.addWidget(left_panel)

        # 添加支持的图片格式说明
        format_label = QLabel(f"支持的图片格式: {', '.join(sorted(SUPPORTED_FORMATS))}")
        format_label.setWordWrap(True)
        left_layout.addWidget(format_label)

        # 模型选择
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
        model_layout.addWidget(QLabel("选择模型:"))
        model_layout.addWidget(self.model_combo)
        left_layout.addLayout(model_layout)

        # 数据集选择
        dataset_layout = QHBoxLayout()
        self.dataset_path_label = QLabel("未选择数据集")
        dataset_layout.addWidget(self.dataset_path_label)
        self.select_dataset_btn = QPushButton("选择数据集文件夹")
        self.select_dataset_btn.clicked.connect(self.select_dataset)
        dataset_layout.addWidget(self.select_dataset_btn)
        left_layout.addLayout(dataset_layout)

        # 类别设置
        class_layout = QVBoxLayout()
        class_layout.addWidget(QLabel("设置检测类别（每行一个类别）:"))
        self.class_edit = QTextEdit()
        self.class_edit.setPlaceholderText("例如：\n人\n猫\n狗")
        self.class_edit.setMaximumHeight(100)
        self.class_edit.textChanged.connect(self.update_class_names)  # 添加文本变化事件
        class_layout.addWidget(self.class_edit)
        left_layout.addLayout(class_layout)

        # 训练参数设置
        params_layout = QHBoxLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        params_layout.addWidget(QLabel("训练轮数:"))
        params_layout.addWidget(self.epochs_spin)
        
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        params_layout.addWidget(QLabel("图像大小:"))
        params_layout.addWidget(self.imgsz_spin)
        left_layout.addLayout(params_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        # 训练按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        left_layout.addWidget(self.train_btn)

        # 右侧标注面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        layout.addWidget(right_panel)

        # 图片列表和标注控制
        list_control_layout = QHBoxLayout()
        
        # 图片列表
        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.addWidget(QLabel("图片列表:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.load_image)
        list_layout.addWidget(self.image_list)
        list_control_layout.addWidget(list_widget, stretch=2)
        
        # 标注控制
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 类别选择
        control_layout.addWidget(QLabel("当前标注类别:"))
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self.update_current_class)
        control_layout.addWidget(self.class_combo)
        
        # 标注操作按钮
        self.clear_btn = QPushButton("清除当前标注")
        self.clear_btn.clicked.connect(self.clear_current_annotation)
        control_layout.addWidget(self.clear_btn)
        
        self.save_btn = QPushButton("保存当前标注")
        self.save_btn.clicked.connect(self.save_current_annotation)
        control_layout.addWidget(self.save_btn)
        
        list_control_layout.addWidget(control_widget, stretch=1)
        right_layout.addLayout(list_control_layout)

        # 图片显示和标注区域
        self.image_label = ImageLabel()
        right_layout.addWidget(self.image_label)

        # 标注说明
        help_text = """
标注说明：
1. 在左侧输入要检测的目标类别
2. 在右侧列表选择要标注的图片
3. 选择当前要标注的类别
4. 在图片上按住鼠标左键拖动来画框
5. 标注完成后点击"保存当前标注"
6. 所有图片都标注完成后即可开始训练
        """
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        right_layout.addWidget(help_label)

        self.dataset_path = None
        self.training_thread = None

    def update_class_names(self):
        """更新类别名称列表"""
        self.class_names = [line.strip() for line in self.class_edit.toPlainText().split('\n') if line.strip()]
        
        # 更新类别选择下拉框
        current_text = self.class_combo.currentText()
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        
        # 尝试恢复之前选择的类别
        index = self.class_combo.findText(current_text)
        if index >= 0:
            self.class_combo.setCurrentIndex(index)

    def update_current_class(self, index):
        """更新当前选择的类别"""
        self.image_label.current_class = index

    def load_image(self, item):
        """加载选中的图片"""
        image_path = item.data(Qt.UserRole)
        self.image_label.setImage(image_path)
        
        # 加载已有的标注
        label_path = self.get_label_path(image_path)
        self.image_label.load_labels(label_path)

    def get_label_path(self, image_path):
        """获取标注文件路径"""
        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_dir = image_dir.replace('images', 'labels')
        return os.path.join(label_dir, base_name + '.txt')

    def clear_current_annotation(self):
        """清除当前图片的标注"""
        self.image_label.clear_boxes()

    def save_current_annotation(self):
        """保存当前图片的标注"""
        if not self.image_label.image_path:
            return
            
        label_path = self.get_label_path(self.image_label.image_path)
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        self.image_label.save_labels(label_path)
        QMessageBox.information(self, "成功", "标注���保存")

    def select_dataset(self):
        """选择数据集文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择数据集文件夹（包含图片的文件夹）",
            "",
            options=QFileDialog.ShowDirsOnly
        )
        
        if folder:
            # 检查是否包含支持的图片格式
            has_images = False
            image_files = []
            for root, _, files in os.walk(folder):
                for file in files:
                    if any(file.lower().endswith(f".{fmt}") for fmt in SUPPORTED_FORMATS):
                        has_images = True
                        image_files.append(os.path.join(root, file))

            if not has_images:
                QMessageBox.warning(self, "错误", f"所选文件夹中没有找到支持的图片格式文件\n支持的格式: {', '.join(sorted(SUPPORTED_FORMATS))}")
                return

            # 显示找到的图片数量
            QMessageBox.information(self, "成功", f"找到 {len(image_files)} 个支持的图片文件")
            
            # 创建数据集目录结构
            train_dir = os.path.join(folder, 'images/train')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(os.path.join(folder, 'labels/train'), exist_ok=True)
            
            # 更新界面
            self.dataset_path = os.path.join(folder, 'dataset.yaml')
            self.dataset_path_label.setText(f"据集: {os.path.basename(folder)}")
            
            # 更新图片列表
            self.image_list.clear()
            for img_file in image_files:
                target_path = os.path.join(train_dir, os.path.basename(img_file))
                if not os.path.exists(target_path):
                    import shutil
                    shutil.copy2(img_file, target_path)
                
                item = QListWidgetItem(os.path.basename(target_path))
                item.setData(Qt.UserRole, target_path)
                self.image_list.addItem(item)

            # 更新类别选择下拉框
            self.class_combo.clear()
            classes = [line.strip() for line in self.class_edit.toPlainText().split('\n') if line.strip()]
            self.class_combo.addItems(classes)

    def create_yaml(self, folder):
        """创建 YAML 配置文件"""
        classes = [line.strip() for line in self.class_edit.toPlainText().split('\n') if line.strip()]
        if not classes:
            QMessageBox.warning(self, "错误", "请至少输入个检测类别")
            return None

        yaml_content = {
            'path': folder,
            'train': 'images/train',
            'val': 'images/train',
            'test': 'images/train',
            'names': {i: name for i, name in enumerate(classes)},
            'nc': len(classes)
        }

        yaml_path = os.path.join(folder, 'dataset.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        return yaml_path

    def start_training(self):
        """开始训练"""
        if not self.dataset_path:
            QMessageBox.warning(self, "错误", "请先选择数据集")
            return

        if not self.class_edit.toPlainText().strip():
            QMessageBox.warning(self, "错误", "请输入检测类别")
            return

        # 查是所有图片都有标注
        train_dir = os.path.join(os.path.dirname(self.dataset_path), 'images/train')
        label_dir = os.path.join(os.path.dirname(self.dataset_path), 'labels/train')
        
        missing_labels = []
        for img_file in os.listdir(train_dir):
            if any(img_file.lower().endswith(f".{fmt}") for fmt in SUPPORTED_FORMATS):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                if not os.path.exists(os.path.join(label_dir, label_file)):
                    missing_labels.append(img_file)
        
        if missing_labels:
            msg = "以下图片还没有标注：\n" + "\n".join(missing_labels)
            reply = QMessageBox.question(self, "警告", msg + "\n\n是否继续训练？",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "警告", "训练正在进行中")
            return

        # 创建 YAML 文件
        folder = os.path.dirname(self.dataset_path)
        yaml_path = self.create_yaml(folder)
        if not yaml_path:
            return

        model_path = os.path.join("weights", self.model_combo.currentText())
        
        self.training_thread = TrainingThread(
            model_path=model_path,
            data_yaml=yaml_path,
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value()
        )
        
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.error.connect(self.training_error)
        
        self.train_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.training_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def training_finished(self):
        self.train_btn.setEnabled(True)
        save_dir = os.path.abspath("runs/detect/train")
        message = f"""训练已完成！

模型保存在：{save_dir}

其中：
- best.pt: 最佳模型
- last.pt: 最后一次保存的模型
- results.csv: 训练结果数据
- confusion_matrix.png: 混淆矩阵
- results.png: 训练过程图表"""
        
        QMessageBox.information(self, "完成", message)

    def training_error(self, error_msg):
        self.train_btn.setEnabled(True)
        QMessageBox.critical(self, "错误", f"训练过程中出现错误：\n{error_msg}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_()) 