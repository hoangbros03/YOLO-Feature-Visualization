# week6-featuresVisualization

This repository allows the user to extract the tensors created in each layer during the forward process before following the steps.
It is made as a task of the 2023 Cinnamon Bootcamp. 

# Instruction

Optional: You can create new isolate environment (using venv or conda) to avoid conflict between package versions

Step 1: Git clones and download pre-trained yolov7 model:
```
git clone https://github.com/hoangbros03/week6-featuresVisualization.git
cd week6-featuresVisualization
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
cd ../
```

Step 2: Install needed dependecies
```
pip install -r requirements.txt
```


Step 3: Run the program
```
python3 image_extraction.py -gpu -p=<model_path> -i=<image_dir> -o=<output_dir> -l=<limit images per layer>
```
_Remember to enable GPU, otherwise you will see the error "RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'". This is caused by yolov7 itself._
