# Color Transfer

Given a composite image obtained by cut-and-paste, we can adjust the foreground color to match the background by using Reinhard's algorithm, which is a traditional color transfer method. When the requirement for the harmonized result is not very high, we can use [traditional color transfer methods](https://github.com/bcmi/Color-Transfer-for-Image-Harmonization) instead of image harmonization methods. When the foreground and background have pure colors and we simply want to match their colors, traditional color transfer methods may work better than image harmonization methods. 

> **Color Transfer between Images**  [[paper]](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)<br>
>
> E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley <br>
> Accepted by **IEEE Computer Graphics and Applications 2001**.

## Brief Method Summary

Reinhard's algorithm achieves color transfer by taking background as source and applying its color characteristic to foreground object. Specifically,  this method adjusts the mean and the standard deviation of L*αβ* channels to match the color distributions between foreground and background. 
