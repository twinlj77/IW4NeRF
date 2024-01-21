from torch import nn #导入对应库
import torch
import torch.nn.functional as F #一个包含卷积函数的库
# 声明体系结构
class encoder_Net(nn.Module):  # 定义一个类torch.nn.Module是所有神经网络模块的基类，所有的神经网络模型都应该继承这个基类
    def __init__(self):  # 初始化函数
        super(encoder_Net, self).__init__()
        self.define_encoder()
        self.define_decoder()

    def define_encoder(self):
        self.encoder_payload_1= nn.Conv2d(3, 32,   kernel_size=3,padding=1)
        self.encoder_payload_2= nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.encoder_source_1 = nn.Conv2d(3, 32,  kernel_size=3,padding=1)
        self.encoder_source_2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.encoder_source_3 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.encoder_source_4 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.encoder_source_5= nn.Conv2d(32, 3, kernel_size=3, padding=1)


    def define_decoder(self):
        self.decoder_layers1 = nn.Conv2d(3, 32,  kernel_size=3, padding=1)
        self.decoder_layers2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.decoder_layers3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.decoder_layers4 = nn.Conv2d(96, 32, kernel_size=3, padding=1)

        # payload_decoder
        self.decoder_payload1 = nn.Conv2d(32, 3,  kernel_size=3,padding=1)


    def forward(self, x):
        source, payload = x
        s = source.contiguous().view((-1, 3,800, 800))
        p = payload.contiguous().view((-1, 3, 800, 800))
        # --------------------------- Encoder -------------------------
        p1 = F.relu(self.encoder_payload_2(p))#3
        #卷积层一
        p = F.relu(self.encoder_payload_1(p))  # 32
        s = F.relu(self.encoder_source_1(s))  # 32
        m1p = p
        m1s = s
        s1 = torch.cat((s, p), 1)  # 64
        #卷积层二
        s = F.relu(self.encoder_source_2(s1))  # 32
        s2 = torch.cat((m1s, m1p, s), 1)  # 96
        s = F.relu(self.encoder_source_3(s2))  # 32
        s3=torch.cat((m1s, m1p, s), 1) #96
        s = F.relu(self.encoder_source_4(s3))#32
        s = F.relu(self.encoder_source_5(s))#3
        #s = s + p1  # 3
        encoder_output = s
        # 注意力机制
       # s = s.contiguous().view(-1, 3, 320, 320)
       # input = s
        #encoder_output1 = model1(input)
        # -------------------- Decoder --------------------------
        d = encoder_output.contiguous().view(-1, 3, 800, 800)
        d1 = F.relu(self.decoder_layers1(d))#32
        d2= F.relu(self.decoder_layers2(d1))#32
        # ---------------- decoder_payload ----------------
        d3 = torch.cat((d1, d2), 1)#64
        d4=F.relu(self.decoder_layers3(d3))#32
        d5 = torch.cat((d1, d2,d4), 1)  # 96
        d6=F.relu(self.decoder_layers4(d5))#32
        decoded_payload = self.decoder_payload1(d6)

        # ---------------- decoder_source ----------------
       # d = F.relu(self.decoder_source1(init_d))
        #d = F.relu(self.decoder_source2(d))
        #d = F.relu(self.decoder_source3(d))
        #d = F.relu(self.decoder_source4(d))
       # d = F.relu(self.decoder_source5(d))
        #d = F.relu(self.decoder_source6(d))
       # d = F.relu(self.decoder_source7(d))
        #d = F.relu(self.decoder_source8(d))

        #ecoded_source = self.decoder_source9(d)

        return encoder_output, decoded_payload#, decoded_source
modelE = encoder_Net()
modelE