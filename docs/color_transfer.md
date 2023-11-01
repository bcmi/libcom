# Color Transfer

Given a composite image obtained by copy-and-paste, we transfer the color of foreground object to fit background scene by using:

**Reinhard's algorithm**:

> **Color Transfer between Images**  [[paper]](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)<br>
>
> E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley <br>
> Accepted by **IEEE Computer Graphics and Applications 2001**.

## Brief Method Summary

Reinhard's algorithm achieves color transfer by taking background as source image and applying its color characteristic to foreground object. Specifically,  this method adjusts the mean and the standard deviation of L*αβ* channels to match the global color distribution of two images. 