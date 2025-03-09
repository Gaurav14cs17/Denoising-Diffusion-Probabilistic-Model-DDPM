---

# Denoising Diffusion Probabilistic Model (DDPM)

![UNet Model](https://github.com/Gaurav14cs17/Denoising-Diffusion-Probabilistic-Model-DDPM/blob/main/unet.png)

This repository contains an implementation of the **Denoising Diffusion Probabilistic Model (DDPM)** for generating high-quality images. The model follows the default ResNet block architecture without any multiplying factors for simplicity. The current implementation uses a **UNet architecture** for the denoising process.

---

## About DDPM
DDPM is a generative model that learns to generate data by gradually denoising a random noise distribution. It is based on the principles of **diffusion processes** and has shown impressive results in image generation tasks.

---

## Model Architecture
The implementation uses a **UNet** architecture with **ResNet blocks** for the denoising process. The UNet is a popular choice for diffusion models due to its ability to capture both local and global features effectively.

### Key Features:
- **ResNet Blocks**: Default ResNet blocks are used without any scaling factors for simplicity.
- **UNet**: The UNet architecture is employed for the denoising network.

---

## Usage

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Gaurav14cs17/Denoising-Diffusion-Probabilistic-Model-DDPM.git
   cd Denoising-Diffusion-Probabilistic-Model-DDPM
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training
To train the model, run the following script:
```bash
python train.py
```

### Inference
To generate images using the trained model, run:
```bash
python generate.py
```

---

## Results
The model generates high-quality images by iteratively denoising a random noise distribution. Below is an example of the UNet architecture used in this implementation:

![UNet Architecture](https://github.com/Gaurav14cs17/Denoising-Diffusion-Probabilistic-Model-DDPM/blob/main/unet.png)

---

## References
This implementation is based on the following resources:
- [Diffusion Models Tutorial by FilippoMB](https://github.com/FilippoMB/Diffusion_models_tutorial/tree/main)

---

## License
This project is open-source and available under the MIT License. For more details, see the [LICENSE](LICENSE) file.

---

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

---

## Acknowledgments
- Thanks to the authors of the original DDPM paper for their groundbreaking work.
- Special thanks to [FilippoMB](https://github.com/FilippoMB) for the tutorial and reference implementation.

---

### Why This Structure Works
1. **Clear and Concise**: Each section is well-defined and easy to follow.
2. **Visual Appeal**: Includes an image of the UNet architecture to make the README more engaging.
3. **User-Focused**: Provides all the information a user needs to get started, including installation, training, and inference instructions.
4. **References**: Gives credit to the original authors and resources used in the implementation.

---

### Next Steps
- **Add Pretrained Models**: Provide pretrained models for users to experiment with.
- **Benchmarking**: Include performance benchmarks and comparisons with other models.
- **Detailed Documentation**: Add more details about the architecture, training process, and hyperparameters.
