import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class UTRDDPMPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        
        self.register_modules(unet=unet, scheduler=scheduler)


    @torch.no_grad()
    def __call__(self, config):
        # initialize noise
        noise = torch.randn((config.current_gen_batch, self.unet.in_channels, self.unet.input_length))
        noise = noise.to(config.device)

        for t in self.progress_bar(self.scheduler.timesteps):
            # predict clean sequences
            pred_sequences = self.unet(noise, t).sample

            noise = self.scheduler.step(pred_sequences, t, noise).prev_sample
        
        return pred_sequences
