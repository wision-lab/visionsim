from torch.optim import AdamW

from .IFNet_HDv3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self, *args, **kwargs):
        self.flownet = IFNet()
        self.device()
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path):
        def convert(param):
            return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}

        if torch.cuda.is_available():
            self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path), weights_only=True)))
        else:
            self.flownet.load_state_dict(convert(torch.load("{}/flownet.pkl".format(path), map_location="cpu", weights_only=True)))

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        flow, mask, merged = self.flownet(imgs, scale_list)
        return merged[2]
