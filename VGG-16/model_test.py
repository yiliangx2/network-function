import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import VGG16

def test_data_process():
    test_data = FashionMNIST(root='./data', train=False, transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]), download=True)
    test_dataloader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)

    return test_dataloader

def test_model_process(model, test_dataloader):
    # 设定测试设备，如果有GPU则使用GPU，否则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播，不计算梯度，以节省内存和计算资源
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将数据放入测试设备中
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            # 设置模型为评估模式
            model.eval()

            # 前向传播
            output = model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    # 计算测试集的准确度
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率:", test_acc)

if __name__ == "__main__":
    # 加载模型
    model = VGG16()
    model.load_state_dict(torch.load('best_model.pth'))
    # 获取测试数据
    test_dataloader = test_data_process()
    # 测试模型
    test_model_process(model, test_dataloader)

    # 打印测试结果
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # with torch.no_grad():
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)

    #         model.eval()
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim=1)
            
    #         result = pre_lab.item()
    #         label = b_y.item()
    #         print("预测结果:", classes[result], "真实标签:", classes[label])
