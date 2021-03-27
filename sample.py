import sys
import torch
from utils import *
from PIL import Image

import concurrent.futures
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Model checkpoints
srgan_checkpoint = "./checkpoint_srgan.pth.tar"
srgan_generator = torch.load(srgan_checkpoint)['generator'].to(device)
srgan_generator.eval()

def visualize_sr(img, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                           Image.BICUBIC)

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    return sr_img_srgan

def up_file(inFile, outFile):
    if os.path.exists(outFile) == False:
        img = visualize_sr(inFile)
        img.save(outFile)

if __name__ == "__main__":
    start_time = time.process_time()
    in_folder = "../in-frames"
    out_folder = "../out-frames"
    (_, _, filenames) = next(os.walk(in_folder))

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for f in filenames:
            in_file = os.path.join(in_folder, f)
            out_file = os.path.join(out_folder, f)
            futures.append(executor.submit(up_file, in_file, out_file))
        for future in concurrent.futures.as_completed(futures):
            future.result()
                
        print(time.process_time() - start_time)
    torch.cuda.empty_cache()