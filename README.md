# Image-Editing-In-Fashion-Industry

This is the repository for the final project of CMU Fall 2024 course Visual Learning and Recognition.

The Fashion-GEN dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/bothin/fashiongen-validation/code).

## Data Preprocessing

After downloading the data from Kaggle, run the following command to extract data into `.pkl` files:

```
python data_preprocessing.py --input_path <Path to .h5 file> --output_path <Path to output pickle file>
```

Then, run the following command to generate images and required `.json` files:

```
python data_conversion.py --input_path <Path to the input pickle file> --output_image_path <Path to the output images directory> --output_finetune_json_path <Path to the output json file for finetuning> --output_image_editing_json_path <Path to the output json file for image editing>
```

The above commands will create an image directory an two `.json` files. One of them is for fine-tuning purpose, and the other is for image editing.
