The example files and the blender scenes the documentation renders live in the shared [`visionsim-assets` folder](https://drive.google.com/drive/folders/1gRxhL3rbGDTfgKytre8WkbBu-QDJFy15?usp=sharing), under `assets`. Using the gdown package (eg: `pip install gdown` or `uv tool install gdown`) you can download them by running the following *inside this examples directory*:

```
gdown --fuzzy --folder https://drive.google.com/drive/folders/16HvbXAYI2vELb9RlBtSvKA9uqhWSAdJg?usp=sharing
```

You should now have an `assets` directory in `visionsim/examples`. From here, you can run examples like so:
```
python blender/render_animation.py 
```
