import numpy as np
import os
import torch as th
from PIL import Image
from MSG_GAN.FID import fid_score
from MSG_GAN.GAN import Generator
from tqdm import tqdm
from torch.backends import cudnn
from scipy.misc import imsave
from shutil import rmtree

cudnn.benchmark = True  # fast mode on
th.manual_seed(3)  # set seed for reproducible calculations
# note that this is not 100 % reproducible as pytorch
# may have different behaviour on different machines

# ====================================================================================
# | Required paramenters
# ====================================================================================
device = th.device("cuda" if th.cuda.is_available() else "cpu")
models_path = "/data/_GENERATED/BMSG-GAN/celebahq_high_big/models"
log_file = "/data/_GENERATED/BMSG-GAN/celebahq_high_big/fid_scores.txt"
real_stats_path = "/data/celebA-HQ_png_1024/real_stats/real_stats.npz"
temp_fid_path = "/data/_GENERATED/BMSG-GAN/celebahq_high_big/temp_fid_samples"
total_range = 220
start = 0  # keeping this 0 for a very obfuscated and weird reason
step = 10
gen_fid_images = 30000
depth = 9
latent_size = 512
batch_size = 16
# ====================================================================================

fid_values = []  # initialized to empty list

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
    model_file_path = os.path.join(models_path, model_file)

    # create a new generator object
    gen = th.nn.DataParallel(
        Generator(depth=depth, latent_size=latent_size).to(device)
    )

    # load these weights into the model
    gen.load_state_dict(th.load(model_file_path))

    # empty the temp directory and make it to ensure it exists
    if os.path.isdir(temp_fid_path):
        rmtree(temp_fid_path)
    os.makedirs(temp_fid_path, exist_ok=True)

    print("\n\nLoaded model:", epoch)
    print("weights loaded from:", model_file_path)
    print("generating %d images using this model ..." % gen_fid_images)
    pbar = tqdm(total=gen_fid_images)
    generated_images = 0
    while generated_images < gen_fid_images:
        b_size = min(batch_size, gen_fid_images - generated_images)
        latents = th.randn(b_size, latent_size)
        latents = ((latents / latents.norm(dim=-1, keepdim=True))
                      * (latent_size ** 0.5))
        imgs = gen(latents)[-1].detach()
        imgs = [adjust_dynamic_range(img) for img in imgs]

        for img in imgs:
            save_file = os.path.join(temp_fid_path, str(generated_images + 1) + ".jpg")
            imsave(save_file, img.permute(1, 2, 0).cpu().numpy())
            generated_images += 1
        pbar.update(b_size)
    pbar.close()

    # Free up resource from pytorch
    del gen

    # Now calculating the fid score for these generated_images ...
    print("Calculating the FID Score for this model ...")
    fid_value = fid_score.calculate_fid_given_paths(
        (real_stats_path, temp_fid_path),
        batch_size,
        True if device == th.device("cuda") else False,
        2048  # using the default value
    )
    print("Obtained FID Score:", fid_value)

    # log the FID score for reference
    with open(log_file, "a") as fil:
        fil.write(str(epoch) + "\t" + str(fid_value) + "\n")
