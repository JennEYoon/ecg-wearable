data loader scripts  

 * peak-centered (single heart beat)  

 * time-series (multiple heart beats)


### PTBXL g1 186 cols, label 187  
Read each file (1000) - remember file name
Source: 12 x 5000, one sample per file  
* For each of 12 leads, 
* downsample to 250 htz
* FFT, Bankpass filter
* scale signals from 0 to 1.0  
 - > save each lead processed
* read diagnostic infor for file
* Combine 12 rows - ouput to processed dir, same file name, what data format?     

New loop
* Read each processed file, 12 leads ( select just II and V5 leads )
* peak center and split & zero pad into 186 rows
* data{0:186), label (187), filename (188), lead(189)
* append to window
* end of file - write 2 rows to data  
* Read next file.  

Sorted (filename), append to file  
  
Source: label from .CSV file with filename, meta data, sorted.  
        Create Label class 4-6 classes.  One-hot encoded or number float?  
        

   
