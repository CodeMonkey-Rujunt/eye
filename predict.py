import torch
import pandas as pd
from PIL import Image
import os
from torchvision import transforms
import torch.nn.functional as F
import numpy as np  # 导入 numpy，用于数值计算
import torch.nn as nn  # 导入神经网络模块
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm


class DualEfficientNet(nn.Module):
    def __init__(self, num_classes=8):
        super(DualEfficientNet, self).__init__()
        # 使用新版 API 加载预训练 EfficientNet_b0 模型
        self.eff_left = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.eff_right = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.eff_left.classifier = nn.Identity()  # 移除左眼模型的分类层
        self.eff_right.classifier = nn.Identity()  # 移除右眼模型的分类层
        self.fc = nn.Linear(1280 * 2, num_classes)  # 全连接层，将左右眼特征拼接后映射到目标类别

    def forward(self, img_left, img_right):
        feat_left = self.eff_left(img_left)  # 获取左眼特征，形状 (B, 1280)
        feat_right = self.eff_right(img_right)  # 获取右眼特征，形状 (B, 1280)
        features = torch.cat([feat_left, feat_right], dim=1)  # 拼接左右眼特征，形状 (B, 2560)
        out = self.fc(features)  # 全连接层输出 logits
        return out  # 返回模型输出
# 定义模型路径和测试图片目录
model_path = 'checkpoints/model_epoch_24.pth'  # 训练好的模型路径
test_images_dir = 'ODIR-5K_Testing_Images'  # 测试图片目录
output_csv_path = 'new_output.csv'  # 输出 CSV 文件路径

# 定义图片预处理方式
test_transform = transforms.Compose([
    transforms.Resize(236),  # 缩放较小边为236，保持宽高比
    transforms.CenterCrop(224),  # 从中心裁剪224x224
    transforms.ToTensor(),  # 转换为张量

])


# 定义加载训练好的模型
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择 GPU（若可用）或 CPU
    model = DualEfficientNet(num_classes=8).to(device)  # 初始化模型并移动至设备

    # 加载模型权重，并确保数据类型匹配
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)  # 再次确保模型在正确的设备上
    model.eval()  # 设置模型为评估模式
    return model


# 预测函数，返回预测结果
def predict(model, left_img_path, right_img_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # 确保模型在正确的设备上

    # 加载并预处理左右眼图像
    left_img = Image.open(left_img_path).convert('RGB')
    right_img = Image.open(right_img_path).convert('RGB')

    left_img = test_transform(left_img).unsqueeze(0).to(device)  # 移动到 GPU 或 CPU
    right_img = test_transform(right_img).unsqueeze(0).to(device)  # 移动到 GPU 或 CPU

    # 将图像输入模型
    with torch.no_grad():
        left_features = model.eff_left(left_img)  # 获取左眼图像的特征
        right_features = model.eff_right(right_img)  # 获取右眼图像的特征
        features = torch.cat((left_features, right_features), dim=1)  # 拼接特征

        # 通过一个全连接层进行分类
        output = model.fc(features)  # 假设模型中有一个全连接层（具体要根据模型结构调整）
        probabilities = torch.sigmoid(output)  # 使用 sigmoid 获取各类的概率值

    return probabilities.squeeze().cpu().numpy()  # 返回概率值


# 处理测试数据并输出结果
def process_test_data(model, test_images_dir, output_csv_path):
    test_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []  # 存储预测结果
    results_dict = {}
    for img_name in tqdm(test_images, desc="Processing Images", unit="image"):
        # 假设文件名格式为：ID_left.jpg 和 ID_right.jpg
        id_name = int(img_name.split('_')[0])
        left_img_path = os.path.join(test_images_dir, f"{id_name}_left.jpg")
        right_img_path = os.path.join(test_images_dir, f"{id_name}_right.jpg")
        # 进行预测
        probabilities = predict(model, left_img_path, right_img_path)
        all_probabilities = (probabilities > 0.5).astype(int)
        # 将预测结果和 ID 存储
        if id_name not in results_dict:
            results_dict[id_name] = all_probabilities
        else:
            # 如果 ID 已经存在，合并左右眼的预测结果
            results_dict[id_name] = np.maximum(results_dict[id_name], all_probabilities)
    # 最终结果
    # 按照 id_name 从小到大排序
    sorted_results = sorted(results_dict.items(), key=lambda x: x[0])
    for id_name, probabilities in sorted_results:
        results.append([id_name] + probabilities.tolist())
    # 创建 DataFrame 并保存为 CSV 文件
    df = pd.DataFrame(results, columns=['ID', 'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    df.to_csv("new_output.csv", index=False)
    print(f"预测结果已保存到 {output_csv_path}")


# 主程序
if __name__ == "__main__":
    # 加载训练好的模型
    model = load_model(model_path)

    # 处理测试数据并生成预测结果
    process_test_data(model, test_images_dir, output_csv_path)
