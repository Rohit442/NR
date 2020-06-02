# Fashion Classification 
Image classification model trained on [Myntra dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small). It can classify **65 categories** from Myntra catalogue. The model is trained on Resnet architecture using Pytorch.

# Demo Application UI

<table>
    <tr>
    <th> Home Page </th>
    <th>Result Page</th>
  </tr>
  <tr>
    <td><img src="/Docs/index_page.png" height=40% ></td>
    <td><img src="/Docs/result_croped.png" width=100% ></td>
  </tr>
  </table>
  
  
# Software packages used
* Pytorch
* Fastai Vision
* Flask Ngrok
* Flask
* Pandas
* Numpy

# Running the application

Run [Demo_app.ipynb](Demo_app.ipynb) notebook either locally or from google Colab. 

# Demo Application
![alt text](/Docs/Demo.gif "Demo")


# Folder Structure
* [model/Analysing_dataset.ipynb](model/Analysing_dataset.ipynb) - Notebook used to analyse dataset.
* [model/Model_creation.ipynb](model/Model_creation.ipynb) - Notebook used to create the model.
* [templates/index.html](templates/index.html) - Home page of UI.
* [templates/result.html](templates/result.html) - Result page of UI.
* [static/upload](static/upload) - Temporary folder used by UI to display output.
* [Training_labels/labels.csv](Training_labels/labels.csv) - Label file to map inference result with label.
* [Training_labels/top_65_catagory.csv](Training_labels/top_65_catagory.csv) - File used to map images with label used for training model.
* [app-ngrok.py](app-ngrok.py) - Script used to run inference to the model via internet URL.
* [app.py](app.py) - Script used to run inference to the model via Localhost URL.
* [Demo_app.ipynb](Demo_app.ipynb) - Demo notebook.
* [Docs/Demo_app.mkv](Docs/Demo_app.mkv) - Demo video.


# Model Interpretation

### Most confused categories:
<pre style="width:750px;height:400px;overflow-x:auto;">
[('Men-Shoes-Footwear', 'Sports Shoes-Footwear', 57),
 ('Shirts-Women-Topwear-Apparel', 'Tshirts-Women-Topwear-Apparel', 54),
 ('Kurtis-Women-Topwear-Apparel', 'Kurtas-Women-Topwear-Apparel', 33),
 ('Sports Shoes-Footwear', 'Men-Shoes-Footwear', 33),
 ('Watches-Women-Watches-Accessories', 'Watches-Men-Watches-Accessories', 33),
 ('Tshirts-Women-Topwear-Apparel', 'Shirts-Women-Topwear-Apparel', 26),
 ('Casual Shoes-Unisex-Shoes-Footwear', 'Men-Shoes-Footwear', 25),
 ('Kurtas-Women-Topwear-Apparel', 'Shirts-Women-Topwear-Apparel', 20),
 ('Shirts-Women-Topwear-Apparel', 'Kurtas-Women-Topwear-Apparel', 20),
 ('Tshirts-Men-Topwear-Apparel', 'Shirts-Men-Topwear-Apparel', 18),
 ('Casual Shoes-Women-Shoes-Footwear', 'Men-Shoes-Footwear', 16),
 ('Casual Shoes-Women-Shoes-Footwear', 'Women-Shoes-Footwear', 16),
 ('Watches-Men-Watches-Accessories', 'Watches-Women-Watches-Accessories', 15),
 ('Watches-Unisex-Watches-Accessories', 'Watches-Men-Watches-Accessories', 15),
 ('Flip Flops-Women-Flip Flops-Footwear',
  'Flip Flops-Men-Flip Flops-Footwear',
  14),
 ('Shirts-Women-Topwear-Apparel', 'Dresses-Girls-Dress-Apparel', 14),
</pre>

### Top losses
![alt text](/Docs/top_loss.png "Top loss")

# Dataset Distribution 
![alt text](/Docs/dataset-distribution.png "dataset distribution.png")

# Potential Effects of the Model
* As training images are all of single person/object; the model work well on single person images without background.
* Used data augmentation on training images like rotating, vertical flipping, lighting and zooming. But still it doesn't covers all the scenorios of a everyday pictures. 
* All the images for inference are downsized to 60x60px and normalized before sending to model. So a high pixel images doesn't make difference.

##### Sample 60x60px image which is normalized


![alt text](/Docs/new_a.jpg "new_a.jpg")


