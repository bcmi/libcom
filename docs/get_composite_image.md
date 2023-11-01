# Get Composite Image

We generate composite image from copy-and-paste followed by image blending, in which a classic image blending algorithm is employed:

**Poisson Blending**:

> **Poisson Image Editing**  [[paper]](https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf)<br>
>
> Patrick Pérez, Michel Gangnet, Andrew Blake <br>
> Accepted by **ACM SIGGRAPH 2003**.

## Brief Method Summary

### Gaussian Blending

Using a Gaussian filter to blur the foreground mask, and applying the blurred mask to combine foreground and background for smoothing the their boundary.    

### Poison Blending

Using generic interpolation machinery based on solving Poisson equations for seamless editing of image regions. Specifically, Poisson image blending enforces the gradient domain consistency with respect to the source image containing the foreground, where the gradient of inserted foreground is computed and propagated from the boundary pixels in the background. 