# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:51:03 2022

@author: Dominic Lightbody
"""
# import os 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="1"



import numpy as np
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import csv
from PIL import Image 
import os

"""Variables to Change"""

"""Change to the unseen version of these files"""
annotations_file = 'validation_set_hidden.csv' #change to private leaderboard "annotations"
path = "CSV_hidden" #point a the full dataset (csv not EDF)

"""Point at the model submitted"""
model_checkpoint = "model"  #path to checkpoint folder where you saved the model (folder not model itself)

"""files this script creates"""
datasetName = "NAME_YOUR_DATASET" #Name the dataset this script will create for inference
predictions_csv = 'NEWNAME.csv' #Name the predictions file output by this script


"""End of Variables to Change"""



list_of_channels = ["F4-C4","F3-C3","C4-T4","C3-T3","C4-Cz","Cz-C3","C4-O2","C3-O1"]

def segment_data(data,N,win_step):
    
    if win_step == 0:
        win_step = N
 
    
    segments = int((N / win_step)*(len(data) / N)) 
    data_ten = np.zeros((segments,N)) 

    for i in range(0,segments):
          
        
        seg_start = i * win_step    
        seg_end = seg_start + N
        
        
        # catch out of bounds errors
        if seg_end > len(data):
            data_ten = data_ten[0:i,:]
            break
        # print(seg_start, seg_end)
        data_ten[i,:] = data[seg_start:seg_end] # ,0
        
        
    print("Data Segmented...")
    
    return data_ten
  
  
#    ***this is RGB****
def saveGASF(im1, im2, im3,classification,a_string):
    
    
    for i in range(0,len(im1)):
        
        
        
        
        zeros = np.zeros((384,384))
        filename = a_string+"_" +str(i) + ".png"
        path = os.path.join(classification,filename) 
        #rgb_uint8 = (np.dstack((X_gasf[i],X_gadf[i],X_MTF[i], X_MTF[i])) * 255.999) .astype(np.uint8)
        rgb_uint8 = (np.dstack((im1[i],im2[i],im3[i])) * 255.999) .astype(np.uint8)
        #rgb_uint8 = (np.dstack((im1[i],zeros,zeros)) * 255.999) .astype(np.uint8)
        im=Image.fromarray(rgb_uint8)
        im = im.save(path)
        #print(i)
        

           




"""###Start of loose code###"""




"""
Create Directories for dataset

"""



#load annotations
annotations = annotations_file
with open(annotations, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    annotations = np.array(list(reader))
    
print(headers)


#create directories for data

index = 0
for i in range(len(annotations)):
    
    
    
    a_string  = annotations[i,1]
    split_string = a_string.split("_", 1)
    substring = split_string[0]
    subject = os.path.join(datasetName,a_string)
    
    if  os.path.isdir(datasetName):
        print("dataset of this name exists")
    else:
        os.mkdir(datasetName)
    
    
    if  os.path.isdir(subject):
        print("subject folder exists")
    else:
        os.mkdir(subject)
        
        
    
        
        
        
    """
    Load each file after its corresponding dir is created / referenced
    """    
    
    from get_data import get_data
    import scipy as sp
    file = a_string + ".csv"
    print(file)
    path = "CSV_hidden"
    path = os.path.join(path, file)
    dataAll = get_data(path)
    
    """
    Segment the data into equal sized segments
    """
    data  = dataAll['F4-C4']
    data2 = dataAll['F3-C3']
    data3 = dataAll['C4-T4']
    a = data
    b = data2
    c = data3
    import pandas as pd;
    import scipy as sp
    
    
    
    
    dec = 25000
    N = 1000
    a = pd.Series(abs(data)**2).rolling(N).mean() **0.5
    a = a.to_numpy()
    a = a[N:]
    a = sp.signal.resample(a,dec)
    
    
    b = pd.Series(abs(data2)**2).rolling(N).mean() **0.5
    b = b.to_numpy()
    b = b[N:]
    b = sp.signal.resample(b,dec)
    
    
    c = pd.Series(abs(data3)**2).rolling(N).mean() **0.5
    c = c.to_numpy()
    c = c[N:]
    c = sp.signal.resample(c,dec)

    

    N = 384 #window size
    win_step = 0 #(int(N * 0.5)) #a
    data_ten = segment_data(a, N, win_step)
    data_ten2 = segment_data(b, N, win_step)  
    data_ten3 = segment_data(c, N, win_step)

    gasf = GramianAngularField( method='summation')
    A_gasf = gasf.fit_transform(data_ten)
    B_gasf = gasf.fit_transform(data_ten2)
    C_gasf = gasf.fit_transform(data_ten3)
    
    print(len(A_gasf))
    saveGASF(A_gasf, B_gasf, C_gasf ,subject,a_string) 
    
"""Submission"""
import os 

from transformers import ConvNextModel, ConvNextConfig, ConvNextFeatureExtractor,ConvNextForImageClassification
from transformers import AutoFeatureExtractor,  SwinForImageClassification, ResNetForImageClassification


"""define feature extractor (important for correct classes)"""
feature_extractor = ConvNextFeatureExtractor.from_pretrained(model_checkpoint)
model = ConvNextForImageClassification.from_pretrained(model_checkpoint)


batch_size = 32 # batch size for training and evaluation


from datasets import load_dataset 


import numpy as np
import os

"""point to val dataset"""
listOfFile = os.listdir(datasetName)

print(listOfFile)



pred_for_file = []
files_in_pred_order = []


annotation = []
pred_per_epoch = []
ground_truth_per_epoch = []
index = 0
votes_per_model = np.zeros((64,4))




from datasets import load_metric

metric = load_metric("accuracy")


from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)   
train_transforms = Compose(
        [
            Resize(feature_extractor.size['shortest_edge']),
            ToTensor(),
            normalize,
        ]
    )

val_transforms = Compose(
        [
            Resize(feature_extractor.size['shortest_edge']),
            ToTensor(),
            normalize,
        ]
    )

def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [
        train_transforms(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch

def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch





votes_per_model = np.zeros((64,4))


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer


model_name = "convnext-384"

args = TrainingArguments(
    f"{model_name}-testing",
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=5e-7,
    #weight_decay=0.1,
    lr_scheduler_type="constant",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=30,
    warmup_ratio=0.0,#01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


import numpy as np

# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    #print("eval_pred: ", eval_pred)
    predictions = np.argmax(eval_pred.predictions, axis=1)

    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}
    



for filename in listOfFile:
    if filename == '.config':
        listOfFile.remove(filename)
        
print(listOfFile)       
counter = 0
for filename in listOfFile:
    
    file = os.path.join(datasetName,filename)
    # option 2: local folder
    print(filename)
    files_in_pred_order.append(filename)
    dataset = load_dataset("imagefolder", data_dir=file)  
    
    
  
    
    # split up training into training + validation
    #splits = dataset["train"].train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    val_ds = dataset["train"]
    
    
    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)   
   
    trainer = Trainer(
        model,
        args,
        train_dataset=train_ds,
        #eval_dataset=val_ds,
        eval_dataset=val_ds,
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    
    train_results = trainer.predict(val_ds).predictions
    
    
    predictions = np.argmax(train_results, axis=1)
    
    res_array = np.array(predictions)
    countgrade_1= np.count_nonzero(res_array == 0)
    countgrade_2= np.count_nonzero(res_array == 1)
    countgrade_3= np.count_nonzero(res_array == 2)
    countgrade_4= np.count_nonzero(res_array == 3)
    
    votes = np.stack((countgrade_1, countgrade_2, countgrade_3, countgrade_4),0)
    votes_per_model[counter,:] += votes
    
    
    most_votes = np.amax(votes)
    
    # if votes[2] > 65 * 0.38:
    #     elected_class = 2
    # else:
        
        
        
    elected_class = np.argmax(votes)
    
    
    pred_for_file.append(elected_class)
    
    file_with_prediction = np.vstack((files_in_pred_order,pred_for_file))
    
    counter+=1
        


# print("Files with predictions:\n", np.transpose(file_with_prediction))   
# print("Votes for this Model:", votes_per_model)


print("Files with predictions:\n", np.transpose(file_with_prediction))   




import csv 
# Example.csv gets created in the current working directory 
with open(predictions_csv, 'w', newline = '') as csvfile:
    my_writer = csv.writer(csvfile, delimiter = ',')
    my_writer.writerow(np.transpose(file_with_prediction))



import csv
import numpy as np

path_temp = predictions_csv
with open(path_temp, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers = next(reader)
    dataRaw = np.array(list(reader)).astype(float)


path_temp2 = annotations_file
with open(path_temp2, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    headers2 = next(reader)
    dataRaw2 = np.array(list(reader))
    


ids = dataRaw2[:,2]
print(ids)




a = "["
b = "]"
c = "'"

idsPred=[]
preds = []
sortedPreds = np.zeros((len(ids),1))

for i in range(0,len(headers)):
    
    for char in a:
        test2 = headers[i].replace(char,"")    
        
    for char in b:
        test2 = test2.replace(char,"")    
    
    for char in c:
        test2 = test2.replace(char,"")  
        
        
        
        
        test = test2.split()
        
        idsPred.append(test[0])
        preds.append(test[1])
                                                                    

        

for i in range(0,len(ids)):
    
    i_d = ids[i]
    
    for j in range(0,len(idsPred)):
        if idsPred[j] == i_d:
            index = j
    
    sortedPreds[i] = int(preds[index]) +1
    
sortedPreds= sortedPreds.astype(int)




predicitions_sorted = "copy."+predictions_csv
with open(predicitions_sorted, "w") as s:
    w = csv.writer(s)
    for row in sortedPreds:
        w.writerow(row)



            




    
    