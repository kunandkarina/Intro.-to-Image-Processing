from PIL import Image
from torchvision import transforms

img = Image.open('girl.png')
img = transforms.Resize((256, 256))(img)
img.save('girl256.png')