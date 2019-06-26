import numpy as np
import os
import torch as th
from PIL import Image
from MSG_GAN.IS.inception_score import get_inception_score
from tqdm import tqdm
from MSG_GAN.GAN import Generator
from torch.backends import cudnn

cudnn.benchmark = True  # fast mode on

th.manual_seed(3)

device = th.device("cuda" if th.cuda.is_available() else "cpu")
files_path = "/data/_GENERATED/BMSG-GAN/flowers_big/models/"
backup_file = "/data/_GENERATED/BMSG-GAN/flowers_big/inception_scores.txt"
total_range = 1110
start = 0
step = 10
tot_images = 5000
depth = 7
latent_size = 512
batch_size = 8

def adjust_dynamic_range(data, drange_in=(-1, 1), drange_out=(0, 1)):
        """
        adjust the dynamic colour range of the given input data
        :param data: input image data
        :param drange_in: original range of input
        :param drange_out: required range of output
        :return: img => colour range adjusted images
        """
        if drange_in != drange_out:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                    np.float32(drange_in[1]) - np.float32(drange_in[0]))
            bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
            data = data * scale + bias
        return th.clamp(data, min=0, max=1)

# go over all the models and calculate it's fid by generating images from that model
for epoch in range((start // step), (total_range // step) + 1):
    epoch = 1 if epoch == 0 else epoch * step
    model_file = "GAN_GEN_SHADOW_" + str(epoch) + ".pth"
    model_file_path = os.path.join(files_path, model_file)

    # create a new generator object
    gen = th.nn.DataParallel(
        Generator(depth = depth, latent_size = latent_size).to(device)
    )

    # load these weights into the model
    gen.load_state_dict(th.load(model_file_path))

    print("\n\nLoaded model:", epoch)
    print("weights loaded from:", model_file_path)
    print("generating %d images using this model ..." % tot_images)
    pbar = tqdm(total=tot_images)
    generated_images = 0
    images = []  # initialized to empty list
    while generated_images < tot_images:
        b_size = min(batch_size, tot_images - generated_images)
        latents = th.randn(b_size, latent_size)
        latents = (latents / latents.norm(dim=-1, keepdim=True)) * (latent_size ** 0.5)
        imgs = gen(latents)[-1].detach()
        images.append(imgs)
        pbar.update(b_size)
        generated_images += b_size
    pbar.close()
    # colour adjust the range of the images
    images = [adjust_dynamic_range(image) for image in images]

    # convert them into an array
    images = th.cat(images, dim=0).cpu().numpy()
    images = images * 255  # change range to [0,255]
    images = images.astype(np.uint8)
    print("Images have been generated with shape:", images.shape)
    print("Image slice: \n%s" % str(images[0, 0, :10, :10]))

    # Free up resource from pytorch
    del gen

    # Now calculating the incpetion score for this ...
    print("Calculating the Inception Score for this model ...")
    inception_score = get_inception_score(images)
    print("Obtained Inception Score:", inception_score)
    # backup the inception score for reference
    with open(backup_file, "a") as fil:
        fil.write(str(epoch) + "\t" + str(inception_score[0]) + "\n")
