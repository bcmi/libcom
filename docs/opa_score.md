# Object Placement Assessment (OPA) Score Prediction

OPA score is to verify whether a composite image is plausible in terms of the object placement, which is predicted by the following method:

**SimOPA**:

> **OPA: Object Placement Assessment Dataset**  [[arxiv]](https://arxiv.org/pdf/2107.01889.pdf) [[homepage]](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA)<br>
>
> Liu Liu, Zhenchen Liu, Bo Zhang, Jiangtong Li, Li Niu, Qingyang Liu, Liqing Zhang<br>

## Brief Method Summary

SimOPA is a binary classifier that is trained to distingush between reasonable and unreasonable object placements. Given a composite image and its foreground mask, SimOPA takes their concatenation as inputs and predicts its rationality score. The score ranges from 0 to 1, where a larger score indicates more reasonable placement. More details about SimOPA can be found in the [homepage](https://github.com/bcmi/Object-Placement-Assessment-Dataset-OPA).