import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor

class DDPMUTRPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):

        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        vae, 
        batch_size = 1,
        generator = None,
        num_inference_steps = 1000,
        output_type = "pil",
        return_dict = True
    ):
    # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            if 'DownBlock1D' in self.unet.config.down_block_types:
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        print(f"Running {num_inference_steps} steps of inference")
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        if not return_dict:
            return (vae.decoder(image),)

        return ImagePipelineOutput(images=image)
