from torchvision import models
import matplotlib.pyplot as plt
import torch
from torchinfo import summary
from captum.attr import Saliency
from captum.attr import Occlusion
from captum.attr import GuidedBackprop
from captum.attr import Deconvolution
from captum.attr import DeepLiftShap
from captum.attr import InputXGradient
from captum.attr import FeaturePermutation
from PIL import Image
from captum.attr import DeepLift
from captum.attr import IntegratedGradients
from torchvision import transforms
torch.manual_seed(42)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 解释器函数
def explainer(model, labels_human, DEVICE):
    model.to(DEVICE)
    model.eval()

    # Explainer
    attribution =InputXGradient(model)

    # 定义图片信息
    image_info = {
        'image1': {'image_path': 'goldfish.png', 'label': 1, 'label_human': 'goldfish'},
        'image2': {'image_path': 'hummingbird.png', 'label': 94, 'label_human': 'hummingbird'},
        'image3': {'image_path': 'black_swan.png', 'label': 100, 'label_human': 'black swan'},
        'image4': {'image_path': 'golden_retriever.png', 'label': 207, 'label_human': 'golden retriever'},
        'image5': {'image_path': 'daisy.png', 'label': 985, 'label_human': 'daisy'}
    }

    fig, ax = plt.subplots(5, 3, figsize=(30, 50))

    for i, (key, value) in enumerate(image_info.items()):
        # 从文件路径加载图片并进行预处理
        image_path = value['image_path']
        image = Image.open(image_path)
        X = preprocess(image).to(DEVICE)

        y = torch.tensor([value['label']]).to(DEVICE)
        label = value['label_human']

        # 预测标签
        output = model(X.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_label = torch.max(probabilities, 0)[1].item()
        # 计算真实标签和预测标签的显著性归因
        attr_true = attribution.attribute(inputs=X.unsqueeze(0), target=y)
        attr_pred = attribution.attribute(inputs=X.unsqueeze(0), target=predicted_label)

        # 将图片转换回原始尺度
        X = X * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)

        # 可视化真实标签和预测标签的显著性归因
        explainer_true, _ = torch.max(attr_true.data.abs(), dim=1)
        explainer_true = explainer_true.cpu().detach().numpy()
        explainer_true = (explainer_true - explainer_true.min()) / (explainer_true.max() - explainer_true.min())

        explainer_pred, _ = torch.max(attr_pred.data.abs(), dim=1)
        explainer_pred = explainer_pred.cpu().detach().numpy()
        explainer_pred = (explainer_pred - explainer_pred.min()) / (explainer_pred.max() - explainer_pred.min())

        ax[i][0].imshow(X[0].permute(1, 2, 0).cpu().numpy())
        ax[i][1].imshow(explainer_true[0])
        ax[i][1].set_title(f"True: {label[0]}", fontsize=48)
        ax[i][2].imshow(explainer_pred[0])
        ax[i][2].set_title(f"Predicted: {labels_human[predicted_label][0]}", fontsize=48)

        for j in range(3):
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])

    fig.subplots_adjust(wspace=0, hspace=0, top=1.0)
    plt.savefig("Saliency.png", bbox_inches='tight')


if __name__ == "__main__":
    # 创建模型
    model_googlenet = models.googlenet(pretrained=True)

    # 模型摘要
    print(summary(model=model_googlenet, input_size=(1, 3, 224, 224), col_width=20,
                  col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'],
                  verbose=0))

    # 加载人类可读标签
    labels_human = {}
    with open(f'imagenet1000_clsidx_to_labels.txt') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().replace("'", "").strip(",")
            if "{" in line or "}" in line:
                continue
            else:
                idx = int(line.split(":")[0])
                lbl = line.split(":")[1].split(",")
                labels_human[idx] = [x.strip() for x in lbl]

    # 运行解释器
    explainer(model_googlenet, labels_human, DEVICE)
