# STMHCPan
## Running Environment
pip install pytorch_gpu==1.10.0  
pip install python==3.7    
pip install scikit-learn==1.0.2   
pip install numpy==1.19.5  
pip install pandas==1.3.5    

hardware: GPU  

## Train and Test  
cd ./code  
You can use STMHCPan_netmhcpan.ipynb for train.  
You can use predict_ensemble.py for predict.   

### For example:  
#### python predict_ensemble.py  
You can change the input file in this .py file

## Data  
Training data sets and independence Test data sets are stored in the data file.

## Result  
The test result data was saved in ./data/test_set  
The picture of test data was saved in ./pic 
