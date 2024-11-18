# TurboEdit for Image Inpainting

This part of the code was modified from [TurboEdit: Text-Based Image Editing Using Few-Step Diffusion Models](https://github.com/GiilDe/turbo-edit?tab=readme-ov-file). I have modified the code so that it could achieve image inpainitng with a given mask image.

Please follow the following steps to install necessary dependencies.

**Note that I recommend using Google Colab to run the code.**

Install packages:

```
pip install bitsandbytes transformers accelerate peft -q
pip install git+https://github.com/huggingface/diffusers.git -q
pip install datasets -q
pip install ml-collections==0.1.1 json-with-comments==1.2.7
(Optional) accelerate config default
```

In addition, make sure to run the following to avoid error:

```
import locale
locale.getpreferredencoding = lambda: "UTF-8"
```

Finally, run the `main.py` file:

```
python main.py --prompts_file="dataset/dataset.json"
```

**Current Issue: Inpainting requires much more steps to actually produce moderate (or poor) results. It will be greatly appreciate if anyone has any good ideas of improving image quality :)**

**Install below items for grounded sam**

```
pip install autodistill==0.1.1
pip install autodistill-grounded-sam==0.1.1
```
