
#importing dependency
from fastai.vision import load_learner,ImageList,DatasetType,open_image,Path
import pandas as pd
import warnings
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
warnings.filterwarnings('ignore')

#config
upload_path = "static/upload"
label_path = 'Training_labels/labels.csv'
model_path = 'model'
threshold = 85



def get_label(predicted_class):
    return df[0].iloc[predicted_class]
    

def print_top_3_pred(preds):
    idx=np.unravel_index(preds.numpy().argsort(axis=None), shape=preds.numpy().shape)
    top_3_pred=idx[0][-3:]
    top_pred=top_3_pred[-1]
    top3_return_msg=""
    for i,val in enumerate(top_3_pred[::-1]):
        top3_return_msg +=str(str(i+1)
                              +"."
                              +str(get_label(val))
                              +" - {:.2f}%\n".format(np.round(preds[val].numpy()*100),2)
                              +"\n")
       
    return top3_return_msg,top_pred

def delete_uploadfiles(upload_path):
    for file in Path(upload_path).glob('*'):
        try:
            file.unlink()
            
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
    return True





#Reading labels
df= pd.read_csv(label_path,header=None)

#assigning model
model= model_path
#learn = load_learner(model)

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = upload_path

#Removing contents in upload folder
delete_uploadfiles(upload_path)

@app.route("/")
def index():
    delete_uploadfiles(upload_path)
    return render_template('index.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        image = request.files['file']
        filename = secure_filename(image.filename)
        
        #saving file in upload path
        image.save(Path(app.config["IMAGE_UPLOADS"]+"/"+ filename))

        my_dict = {}
        #loading images from upload path      
        img_list_loader = ImageList.from_folder(upload_path)
        
        #Checking if valid images are uploaded
        if len(img_list_loader.items)>0:
            #loading model
            load_model = load_learner(model, 
                                  test=img_list_loader)
            #running inference
            preds,y = load_model.get_preds(ds_type=DatasetType.Test)
            index =0
            
            #Processing results for UI
            for preds,img_src in zip(preds,img_list_loader.items):

                top3_return_msg,top_pred = print_top_3_pred(preds)
                
                if(np.round(preds[top_pred].numpy()*100,2)<threshold):
                    custom_msg = "NA"
                    Prediction_percent = "NA"
                else:
                    custom_msg= str(get_label(int(top_pred)))
                    Prediction_percent = str("{:.2f}%".format(np.round(preds[top_pred].numpy()*100,2)))

                temp_val=[]
                temp_val.append(img_src)
                temp_val.append(custom_msg)
                temp_val.append(Prediction_percent)
                temp_val.append(top3_return_msg)

                my_dict[index]=temp_val
                index+=1

            return render_template('result.html', mydict=my_dict)

            
        elif len(img_list_loader.items)== 0:
            return "ERROR: Invalid image. Go back to upload new image"

        

if __name__ == "__main__":
  app.run(debug=False,host='0.0.0.0')

