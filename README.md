# tf2_ganomaly

This repository contains Tensorflow 2.0 implementation of the paper **GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training** [[1]](#Reference), and highly reference on **Pytorch implementation** [[2]](#Reference) from the author of paper.

## Environment

### docker image

includes tensorflow2, sklearn, tqdm, yapf

```bash
docker pull jaycase/tf2:latest
```

## Train and Evaluate

- [mnist.ipynb](https://github.com/chychen/tf2_ganomaly/blob/master/mnist.ipynb)

### TODO

- `uba.ipynb`
- `ffob.ipynb`

## Reference

- [1] Akcay S., Atapour-Abarghouei A., Breckon T.P. (2019) GANomaly: Semi-supervised Anomaly Detection via Adversarial Training. In: Jawahar C., Li H., Mori G., Schindler K. (eds) Computer Vision – ACCV 2018. ACCV 2018. Lecture Notes in Computer Science, vol 11363. Springer, Cham.
- [2] https://github.com/samet-akcay/ganomaly
