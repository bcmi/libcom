# Get Composite Image

We generate a composite image based on foreground and background, by using traditional image blending methods (alpha blending or Poisson blending), which are simple and effective. Poisson blending is the following method. Note that when using Poisson blending, the background color may seep into the foreground in an unexpected way. 

> **Poisson Image Editing**  [[paper]](https://dl.acm.org/doi/abs/10.1145/3596711.3596772)<br>
>
> Patrick Pérez, Michel Gangnet, Andrew Blake <br>
> Accepted by **ACM SIGGRAPH 2003**.

## Brief Method Summary

### Alpha Blending

Alpha blending uses a Gaussian filter to blur the foreground mask, and applies the blurred mask to combine foreground and background to smoothen the boundary.

### Poison Blending

Poison blending solves Poisson equations for seamless image blending. Specifically, Poisson image blending enforces the gradient domain consistency with the source image containing the foreground, where the gradient of inserted foreground is computed and propagated from the boundary pixels in the background. 
