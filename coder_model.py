from torch import nn #导入对应库
import torch
import torch.nn.functional as F #一个包含卷积函数的库
# 声明体系结构
class StegNet(nn.Module):  # 定义一个类torch.nn.Module是所有神经网络模块的基类，所有的神经网络模型都应该继承这个基类
    def __init__(self):  # 初始化函数
        super(StegNet, self).__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        # layer1
        # 格式一般定义为： [b,c,h,w] 其中𝑏表示输入的数量，𝑐表示特征图的通道数,h、w分布表示特征图的高宽
        self.encoder_payload_1 = nn.Linear(3, 32)
        # 输入通道数为1，输出通道为32，卷积核大小3*3，图像填充为1，填充值为0，padding后图像大小变成34*34
        self.encoder_source_1 = nn.Linear(3, 32)

        # layer2
        self.encoder_payload_2 = nn.Linear(32, 32)
        self.encoder_source_2 = nn.Linear(64, 64)
        self.encoder_source_21 = nn.Linear(64, 32)
        #         self.encoder_bn2 = nn.BatchNorm2d(32)对输入的四维数组进行批量标准化处理
        # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)

        # layer3
        self.encoder_payload_3 = nn.Linear(32, 32)
        self.encoder_source_3 = nn.Linear(32, 32)

        # layer4
        self.encoder_payload_4 = nn.Linear(32, 32)
        self.encoder_source_4 = nn.Linear(128, 64)
        self.encoder_source_41 = nn.Linear(64, 32)

        #         self.encoder_bn4 = nn.BatchNorm2d(32)

        # layer5
        self.encoder_payload_5 = nn.Linear(32, 32)
        self.encoder_source_5 = nn.Linear(32, 32)

        # layer6
        self.encoder_payload_6 = nn.Linear(32, 32)
        self.encoder_source_6 = nn.Linear(192, 128)
        self.encoder_source_61 = nn.Linear(128, 64)
        self.encoder_source_62 = nn.Linear(64, 32)

        #         self.encoder_bn6 = nn.BatchNorm2d(32)

        # layer7
        self.encoder_payload_7 = nn.Linear(32, 32)
        self.encoder_source_7 = nn.Linear(32, 32)

        # layer8
        self.encoder_payload_8 = nn.Linear(32, 32)
        self.encoder_source_8 = nn.Linear(256, 128)
        self.encoder_source_81 = nn.Linear(128, 64)
        self.encoder_source_82 = nn.Linear(64, 32)

        #         self.encoder_bn8 = nn.BatchNorm2d(32)

        # layer9
        self.encoder_source_9 = nn.Linear(32, 16, kernel_size=1)

        # layer10
        self.encoder_source_10 = nn.Linear(16, 8, kernel_size=1)

        # layer11
        self.encoder_source_11 = nn.Linear(8, 3, kernel_size=1)

    def define_decoder(self):
        self.decoder_layers1 = nn.Linear(3, 256)
        self.decoder_layers2 = nn.Linear(256, 128)
        #         self.decoder_bn2 = nn.BatchNorm2d(64)

        self.decoder_layers3 = nn.Linear(128, 64)
        self.decoder_layers4 = nn.Linear(64, 64)
        #         self.decoder_bn4 = nn.BatchNorm2d(32)

        self.decoder_layers5 = nn.Linear(64, 32)

        # payload_decoder
        self.decoder_payload1 = nn.Linear(32, 16)
        self.decoder_payload2 = nn.Linear(16, 16)

        self.decoder_payload3 = nn.Linear(16, 8)
        self.decoder_payload4 = nn.Linear(8, 8)

        self.decoder_payload5 = nn.Linear(8, 3)
        self.decoder_payload6 = nn.Linear(3, 3)

        # source_decoder
        self.decoder_source1 = nn.Linear(32, 16)
        self.decoder_source2 = nn.Linear(16, 16)

        self.decoder_source3 = nn.Linear(16, 8)
        self.decoder_source4 = nn.Linear(8, 8)

        self.decoder_source5 = nn.Linear(8, 3)
        self.decoder_source6 = nn.Linear(3, 3)

    def forward(self, x):
        # 定义前向传播
        source, payload = x
        # 特殊用法：参数-1 (自动调整size)view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
        s = source.contiguous().view((-1, 3, 800, 800))

        p = payload.contiguous().view((-1, 3, 800, 800))

        # --------------------------- Encoder -------------------------
        # layer1
        p = F.relu(self.encoder_payload_1(p))
        s = F.relu(self.encoder_source_1(s))

        # layer2
        p = F.relu(self.encoder_payload_2(p))
        s1 = torch.cat((s, p), 1)  # 64
        # torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
        # 使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
        # C = torch.cat( (A,B),0 )  #按维数0拼接（竖着拼）
        # C = torch.cat( (A,B),1 )  #按维数1拼接（横着拼
        s = F.relu(self.encoder_source_2(s1))
        s = F.relu(self.encoder_source_21(s1))
        #         s = self.encoder_bn2(s)

        # layer3
        p = F.relu(self.encoder_payload_3(p))
        s = F.relu(self.encoder_source_3(s))

        # layer4
        p = F.relu(self.encoder_payload_4(p))
        s2 = torch.cat((s, p, s1), 1)  # 128
        s = F.relu(self.encoder_source_4(s2))
        s = F.relu(self.encoder_source_41(s))
        #         s = self.encoder_bn4(s)

        # layer5
        p = F.relu(self.encoder_payload_5(p))
        s = F.relu(self.encoder_source_5(s))

        # layer6
        p = F.relu(self.encoder_payload_6(p))
        s3 = torch.cat((s, p, s2), 1)  # 192
        s = F.relu(self.encoder_source_6(s3))
        s = F.relu(self.encoder_source_61(s))
        s = F.relu(self.encoder_source_62(s))
        #         s = self.encoder_bn6(s)

        # layer7
        p = F.relu(self.encoder_payload_7(p))
        s = F.relu(self.encoder_source_7(s))

        # layer8
        p = F.relu(self.encoder_payload_8(p))
        s4 = torch.cat((s, p, s3), 1)
        s = F.relu(self.encoder_source_8(s4))
        s = F.relu(self.encoder_source_81(s))
        s = F.relu(self.encoder_source_82(s))
        #         s = self.encoder_bn8(s)

        # layer9
        s = F.relu(self.encoder_source_9(s))

        # layer10
        s = F.relu(self.encoder_source_10(s))

        # layer11
        encoder_output = self.encoder_source_11(s)

        # -------------------- Decoder --------------------------

        d = encoder_output.contiguous().view(-1, 3, 800, 800)

        # layer1
        d = F.relu(self.decoder_layers1(d))
        d = F.relu(self.decoder_layers2(d))
        #         d = self.decoder_bn2(d)

        # layer3
        d = F.relu(self.decoder_layers3(d))
        d = F.relu(self.decoder_layers4(d))
        #         d = self.decoder_bn4(d)

        init_d = F.relu(self.decoder_layers5(d))

        # ---------------- decoder_payload ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_payload1(init_d))
        d = F.relu(self.decoder_payload2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_payload3(d))
        d = F.relu(self.decoder_payload4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_payload5(d))
        decoded_payload = self.decoder_payload6(d)

        # ---------------- decoder_source ----------------

        # layer 1 & 2
        d = F.relu(self.decoder_source1(init_d))
        d = F.relu(self.decoder_source2(d))
        # layer 3 & 4
        d = F.relu(self.decoder_source3(d))
        d = F.relu(self.decoder_source4(d))
        # layer 5 & 6
        d = F.relu(self.decoder_source5(d))
        decoded_source = self.decoder_source6(d)

        return encoder_output, decoded_payload, decoded_source


model = StegNet()
model.cuda()