# Tips on GAN and optimization for superres
## Architecture
### SRGAN
- Can train generator only, then GAN, then generator only (while loading models in between)
- After GAN training, can save its model then use generator model only to perform SR and use metrics

### To explore
- Use Conv2DTranspose instead of UpSampling2D
- Scale all pixel values to the range [-1, 1]
- Try LapSRN (good for x8)
- Try SREdgeNET (different concept for SR)
- SRDenseNet https://towardsdatascience.com/review-srdensenet-densenet-for-sr-super-resolution-cbee599de7e8

## Optimization of network
- Tips and tricks to make GANs work https://github.com/soumith/ganhacks

## Ideas
- Deconvolution and Checkerboard Artifacts https://distill.pub/2016/deconv-checkerboard/



