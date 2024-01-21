#Copyright Protection of Neural Radiation Fields Using Invertible Neural Network Watermarking Stega4NeRF

This is the official code for "Copyright Protection of Neural Radiation Fields Using Invertible Neural Network Watermarking"

# Running the code

This code requires Python 3. 8

To train a INN:

```
python train-INN.py 
```

After training for 300 iterations (~8 hours on a single 2080 Ti), you can find the following model at `model.pt`.

---
The training image for NeRF will be used for watermark embedding using INN
```
python INNtest,encoder.py

---
To train a low-res `lego` NeRF:
```
python run_nerf.py --config configs/lego.txt
```
After training for 50k iterations (~3 hours on a single 2080 Ti)
---
The rendered image is processed using the Image Quality Enhancement module.
---
python 恢复网络.py
---
The processed image is then used to extract the watermark using INN
---
python INNtest,decoder.py

---
# Acknowledgements

[NeRF](https://paperswithcode.com/paper/nerf-representing-scenes-as-neural-radiance#code) models are used to implement Copyright Protection of Neural Radiation Fields Using Invertible Neural Network Watermarking. 
