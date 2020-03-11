import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

img = Image.open('/Users/leftthomas/Downloads/a.jpg').convert('L')
tensor = ToTensor()(img).squeeze(0)
u, s, v = torch.svd(tensor)
# s[500:] = 0
dig = torch.diag(s)
a = torch.chain_matmul(u, dig, v.t().contiguous())
img = ToPILImage()(a.unsqueeze(0))
img.save('/Users/leftthomas/Downloads/b.jpg')
