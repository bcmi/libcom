# Image Harmonization

Image harmonization aims to adjust the foreground illumination of a composite image so that it is consistent with the background. We propose a method called PCTNet, which is a color-to-color transformation approach that learns a color mapping to transform the composite foreground. This type of method is efficient and applicable to images of arbitrary resolution. Unlike traditional global color mapping, PCTNet learns local color mapping, meaning it adaptively applies different color transformations to different local regions, resulting in more detailed and natural harmonization effects.

**PCTNet**:

> **PCT-Net: Full Resolution Image Harmonization Using Pixel-Wise Color
Transformations**  [[pdf]](https://openaccess.thecvf.com/content/CVPR2023/papers/Guerreiro_PCT-Net_Full_Resolution_Image_Harmonization_Using_Pixel-Wise_Color_Transformations_CVPR_2023_paper.pdf) [[code]](https://github.com/rakutentech/PCT-Net-Image-Harmonization)<br>
>
> Guerreiro, Julian Jorge Andrade and Nakazawa, Mitsuru and Stenger, Bj\"orn<br>
> Accepted by **CVPR2023**.

## Brief Method Summary

![image_harmonization_PCTNet](../resources/image_harmonization_PCTNet.jpg)


PCTNet takes in a downsampled image and outputs spatial-aware color transformation parameters,  which are interpolated and applied to the foreground region of full-resolution composite image. 
