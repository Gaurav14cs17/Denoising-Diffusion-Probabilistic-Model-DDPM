import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from pathlib import Path

def broadcast(values, broadcast_to):
    values = values.flatten()
    while len(values.shape) < len(broadcast_to.shape):
        values = values.unsqueeze(-1)
    return values


def postprocess(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images


def create_images_grid(images, rows, cols):
    images = [Image.fromarray(image) for image in images]
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def create_sampling_animation(model, pipeline, config, interval=5, every_nth_image=1, rows=2, cols=3):
    noisy_sample = torch.randn(config.eval_batch_size,config.image_channels,config.image_size,config.image_size).to(config.device)
    images = pipeline.sampling(model, noisy_sample, device=config.device, save_all_steps=True)
    fig = plt.figure()
    ims = []
    for i in range(0, pipeline.num_timesteps, every_nth_image):
        imgs = postprocess(images[i])
        image_grid = create_images_grid(imgs, rows=rows, cols=cols)
        im = plt.imshow(image_grid, animated=True)
        ims.append([im])
    plt.axis('off')
    animate = animation.ArtistAnimation(fig, ims, interval=interval, blit=True, repeat_delay=5000)
    path_to_save_animation = Path(config.output_dir, "samples", "diffusion.gif")
    animate.save(str(path_to_save_animation))



class DDPMPipeline:
    def __init__(self, beta_start=1e-4, beta_end=1e-2, num_timesteps=1000):
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_hat = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = num_timesteps

    def forward_diffusion(self, images, timesteps) -> tuple[torch.Tensor, torch.Tensor]:
        gaussian_noise = torch.randn(images.shape).to(images.device)
        alpha_hat = self.alphas_hat[timesteps].to(images.device)
        alpha_hat = broadcast(alpha_hat, images)
        return torch.sqrt(alpha_hat) * images + torch.sqrt(1 - alpha_hat) * gaussian_noise, gaussian_noise

    def reverse_diffusion(self, model, noisy_images, timesteps):
        predicted_noise = model(noisy_images, timesteps)
        return predicted_noise

    @torch.no_grad()
    def sampling(self, model, initial_noise, device, save_all_steps=False):
        image = initial_noise
        images = []
        for timestep in tqdm(range(self.num_timesteps - 1, -1, -1)):
            ts = timestep * torch.ones(image.shape[0], dtype=torch.long, device=device)
            predicted_noise = model(image, ts)
            beta_t = self.betas[timestep].to(device)
            alpha_t = self.alphas[timestep].to(device)
            alpha_hat = self.alphas_hat[timestep].to(device)
            alpha_hat_prev = self.alphas_hat[timestep - 1].to(device)
            beta_t_hat = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta_t
            variance = torch.sqrt(beta_t_hat) * torch.randn(image.shape).to(device) if timestep > 0 else 0
            image = torch.pow(alpha_t, -0.5) * (image -beta_t / torch.sqrt((1 - alpha_hat_prev)) *predicted_noise) + variance
            if save_all_steps:
                images.append(image.cpu())
        return images if save_all_steps else image
