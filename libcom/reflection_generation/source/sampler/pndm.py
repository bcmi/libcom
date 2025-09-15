import torch
import numpy as np
from tqdm.auto import tqdm
import cv2

from .scheduling_pndm import PNDMScheduler

class PNDMSampler(object):
    def __init__(self, model, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.num_timesteps = model.num_timesteps
        self.scheduler = PNDMScheduler(
                                  beta_end=0.012,
                                  beta_schedule='scaled_linear',
                                  beta_start=0.00085,
                                  num_train_timesteps=1000,
                                  set_alpha_to_one=False,
                                  skip_prk_steps=True,
                                  steps_offset=1,
                               )

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               features_adapter=None,
               append_to_context=None,
               cond_tau=0.4,
               style_cond_tau=1.0,
               input=None,
               strength=0.7,
               **kwargs
               ):
        num_inference_steps = S
        C, H, W = shape
        size = (batch_size, C, H, W)
        # print(f'Data shape for PNDM sampling is {size}, eta {eta}')
        device = self.model.betas.device
        self.scheduler.set_timesteps(num_inference_steps)

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]
        num_inference_steps=num_inference_steps - t_start

        if input is None:
            image = torch.randn(size, device=device)
        else:
            # add noise to the composite image
            x0 = self.model.encode_first_stage((input).to(self.model.device))
            x0 = self.model.get_first_stage_encoding(x0)
            noise = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype)
            latent_timestep = timesteps[:1].repeat(batch_size)
            image = self.scheduler.add_noise(original_samples=x0, noise=noise, timesteps=latent_timestep)


        # with tqdm(total=num_inference_steps) as progress_bar:
        for _, t in enumerate(timesteps):
            ts = torch.full((batch_size,), t, device=device)
            image_input = self.scheduler.scale_model_input(image, t)
            residual = self.model.apply_model(image_input, 
                                                ts, 
                                                conditioning)
            image = self.scheduler.step(residual, t, image).prev_sample
                # progress_bar.update()

        return image, 0