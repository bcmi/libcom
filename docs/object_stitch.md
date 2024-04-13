# Object Stitch

Object Stitch is another generative image composition model which can adjust the pose and view of foreground, but the fidelity of generated foreground is worse than ControlComModel.

> **ObjectStitch: Object Compositing with Diffusion Model**  [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_ObjectStitch_Object_Compositing_With_Diffusion_Model_CVPR_2023_paper.pdf) [[code]](https://github.com/bcmi/ObjectStitch-Image-Composition)<br>
>
> Yizhi Song, Zhifei Zhang,  Zhe Lin, Scott Cohen, Brian Price, Jianming Zhang, Soo Ye Kim, Daniel Aliaga<br>
> Accepted by **CVPR 2023**.


## Brief Method Summary

### ![fos_score_FOSE](../resources/objectstitch.jpg)

The framework of Object Stitch consists of a content adaptor and a generator (a pretrained text-to-image diffusion model). The input subject is fed into a ViT and the adaptor which produces a descriptive embedding. At the same time the background image is taken as input by the diffusion model. At each iteration during the denoising stage, we apply the mask on the generated image, so that the generator only denoises the masked area.

Following ObjectStitch, our implementation is based on [Paint-by-Example](https://github.com/Fantasy-Studio/Paint-by-Example), utilizing masked foreground images and employing all class and patch tokens from the foreground image as conditional embeddings. The content adapter in ObjectStitch is implemented using five stacked Transformer blocks in our codebase. We adopt a similar data preparation pipeline to generate and augment training data.