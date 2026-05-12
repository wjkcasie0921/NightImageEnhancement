import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv2d_0 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_0 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_1 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=12, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_1 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_2 = nn.Conv2d(bias=False, dilation=(1,1), groups=12, in_channels=12, kernel_size=(3,3), out_channels=12, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_2 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_3 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=24, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_3 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_4 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=12, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_4 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_5 = nn.Conv2d(bias=False, dilation=(1,1), groups=12, in_channels=12, kernel_size=(3,3), out_channels=12, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_5 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_6 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=24, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_6 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_7 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=12, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_7 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_8 = nn.Conv2d(bias=False, dilation=(1,1), groups=12, in_channels=12, kernel_size=(3,3), out_channels=12, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_8 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=12, num_groups=6)
        self.conv2d_9 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=24, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.gn_9 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_10 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=24, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_10 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_12 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=1, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.conv2d_11 = nn.Conv2d(bias=False, dilation=(1,1), groups=1, in_channels=30, kernel_size=(3,3), out_channels=24, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.gn_11 = nn.GroupNorm(affine=True, eps=1.000000e-05, num_channels=24, num_groups=8)
        self.conv2d_13 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=24, kernel_size=(1,1), out_channels=3, padding=(0,0), padding_mode='zeros', stride=(1,1))

        archive = zipfile.ZipFile('E:\somethings\paper\NightImageEnhancement\mobile\model_conversion\retinex_lite.pnnx.bin', 'r')
        self.conv2d_0.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_0.weight', (24,3,3,3), 'float32')
        self.gn_0.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_0.bias', (24), 'float32')
        self.gn_0.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_0.weight', (24), 'float32')
        self.conv2d_1.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_1.weight', (12,24,1,1), 'float32')
        self.gn_1.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_1.bias', (12), 'float32')
        self.gn_1.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_1.weight', (12), 'float32')
        self.conv2d_2.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_2.weight', (12,1,3,3), 'float32')
        self.gn_2.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_2.bias', (12), 'float32')
        self.gn_2.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_2.weight', (12), 'float32')
        self.conv2d_3.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_3.weight', (24,24,1,1), 'float32')
        self.gn_3.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_3.bias', (24), 'float32')
        self.gn_3.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_3.weight', (24), 'float32')
        self.conv2d_4.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_4.weight', (12,24,1,1), 'float32')
        self.gn_4.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_4.bias', (12), 'float32')
        self.gn_4.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_4.weight', (12), 'float32')
        self.conv2d_5.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_5.weight', (12,1,3,3), 'float32')
        self.gn_5.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_5.bias', (12), 'float32')
        self.gn_5.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_5.weight', (12), 'float32')
        self.conv2d_6.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_6.weight', (24,24,1,1), 'float32')
        self.gn_6.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_6.bias', (24), 'float32')
        self.gn_6.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_6.weight', (24), 'float32')
        self.conv2d_7.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_7.weight', (12,24,1,1), 'float32')
        self.gn_7.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_7.bias', (12), 'float32')
        self.gn_7.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_7.weight', (12), 'float32')
        self.conv2d_8.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_8.weight', (12,1,3,3), 'float32')
        self.gn_8.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_8.bias', (12), 'float32')
        self.gn_8.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_8.weight', (12), 'float32')
        self.conv2d_9.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_9.weight', (24,24,1,1), 'float32')
        self.gn_9.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_9.bias', (24), 'float32')
        self.gn_9.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_9.weight', (24), 'float32')
        self.conv2d_10.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_10.weight', (24,24,3,3), 'float32')
        self.gn_10.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_10.bias', (24), 'float32')
        self.gn_10.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_10.weight', (24), 'float32')
        self.conv2d_12.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.bias', (1), 'float32')
        self.conv2d_12.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_12.weight', (1,24,1,1), 'float32')
        self.conv2d_11.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_11.weight', (24,30,3,3), 'float32')
        self.gn_11.bias = self.load_pnnx_bin_as_parameter(archive, 'gn_11.bias', (24), 'float32')
        self.gn_11.weight = self.load_pnnx_bin_as_parameter(archive, 'gn_11.weight', (24), 'float32')
        self.conv2d_13.bias = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.bias', (3), 'float32')
        self.conv2d_13.weight = self.load_pnnx_bin_as_parameter(archive, 'conv2d_13.weight', (3,24,1,1), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.conv2d_0(v_0)
        v_2 = self.gn_0(v_1)
        v_3 = F.silu(v_2)
        v_4 = self.conv2d_1(v_3)
        v_5 = self.gn_1(v_4)
        v_6 = F.silu(v_5)
        v_7 = self.conv2d_2(v_6)
        v_8 = self.gn_2(v_7)
        v_9 = F.silu(v_8)
        v_10 = torch.cat((v_6, v_9), dim=1)
        v_11 = self.conv2d_3(v_10)
        v_12 = self.gn_3(v_11)
        v_13 = (v_12 + v_3)
        v_14 = F.silu(v_13)
        v_15 = self.conv2d_4(v_14)
        v_16 = self.gn_4(v_15)
        v_17 = F.silu(v_16)
        v_18 = self.conv2d_5(v_17)
        v_19 = self.gn_5(v_18)
        v_20 = F.silu(v_19)
        v_21 = torch.cat((v_17, v_20), dim=1)
        v_22 = self.conv2d_6(v_21)
        v_23 = self.gn_6(v_22)
        v_24 = (v_23 + v_14)
        v_25 = F.silu(v_24)
        v_26 = self.conv2d_7(v_25)
        v_27 = self.gn_7(v_26)
        v_28 = F.silu(v_27)
        v_29 = self.conv2d_8(v_28)
        v_30 = self.gn_8(v_29)
        v_31 = F.silu(v_30)
        v_32 = torch.cat((v_28, v_31), dim=1)
        v_33 = self.conv2d_9(v_32)
        v_34 = self.gn_9(v_33)
        v_35 = (v_34 + v_25)
        v_36 = F.silu(v_35)
        v_37 = self.conv2d_10(v_36)
        v_38 = self.gn_10(v_37)
        v_39 = F.silu(v_38)
        v_40 = self.conv2d_12(v_39)
        v_41 = F.sigmoid(v_40)
        v_42 = ((v_41 * 1.15) + 0.05)
        v_43 = v_42.expand(-1, -1, -1, -1)
        v_44 = torch.tile(v_43, dims=(1,3,1,1))
        v_45 = (v_0 / (v_44 + 0.0001))
        v_46 = torch.clamp(v_45, max=1.0, min=0.0)
        v_47 = torch.cat((v_36, v_0, v_46), dim=1)
        v_48 = self.conv2d_11(v_47)
        v_49 = self.gn_11(v_48)
        v_50 = F.silu(v_49)
        v_51 = self.conv2d_13(v_50)
        v_52 = F.tanh(v_51)
        v_53 = (v_46 + (v_52 * 0.1))
        v_54 = torch.clamp(v_53, max=1.0, min=0.0)
        return v_42, v_54

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("E:\somethings\paper\NightImageEnhancement\mobile\model_conversion\retinex_lite_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.float)

    torch.onnx.export(net, v_0, "E:\somethings\paper\NightImageEnhancement\mobile\model_conversion\retinex_lite_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0', 'out1'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.float)

    import pnnx
    pnnx.export(net, "E:\somethings\paper\NightImageEnhancement\mobile\model_conversion\retinex_lite_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 320, 320, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
