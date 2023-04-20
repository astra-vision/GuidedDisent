This project is about training GAN networks for **image-to-image translation** while keeping into account the **entanglement effects due to occluders**. You can find more information about our work in [our paper](https://arxiv.org/abs/2107.14229). 

In the **specific case of rain**, we use a **realistic drop model** that takes into account parameters related to **drops size, shape, and probability to appear**. 

This is a **demo** that showcases our inference results, in which we trained a GAN only on rainy data, hence having both **wetness and raindrops on the lens**, and we were able to learn to **render wetness only, while rendering raindrops with our physical model**.
