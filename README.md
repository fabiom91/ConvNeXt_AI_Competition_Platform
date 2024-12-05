README for SUBMISSION:

##### FILES SUBMITTED #####

- generateSubmission.py (Creates Validation set based on Unseen version of validation_set.csv, infers on this dataset and outputs a "copy.NAME_OF_FILE.csv". This csv you can simply paste into the unseen validation set class predictions field)
- get_data.py (Helper script for generateSubmission, is called within generateSubmission to simply filter the EEG data)
- model (Folder containing the model. simply point at the FOLDER in generateSubmission.py. Please note point at the FOLDER not a .pt or .pth etc.)
- README.txt (You're Reading it :)
- validation_set.csv (Included with the competition, just here to give context to what is meant by "validation_set.csv" when we referece it)


##### HOW TO RUN THE CODE #####

- install prerequisite libraries (all found at the top of generateSubmission)
- They should all be pip installable

1) Within generateSubmission.py there are 4 variables which will require renaming:
	-annotations_file : this is the path to the unseen version of "validiation_set.csv"
			    This script requires the unseen validation_set.csv to be in the same 				    format as the original supplied by the competition
	
	-path : 	    This path to the folder containing the unseen dataset. Please note this is 				    the CSV version not EDF. Dont need to extract the individual .csv.xz 				    within said folder as the script handles this

	-model_checkpoint: path to the "model" folder supplied in the submission. Please note change 				   this variable to the path of wherever "model" was placed, not any file 				   within this folder.

	-datasetName: 	   Name of the dataset this script will create based on the unseen dataset 				   (you can choose this)

	-predicitons_csv:  Name of the predicitons file the script will create
			   (you can choose this, NOTE: the final predicitions will be in a file named 				   copy.WHATEVER_NAME_YOU_CHOSE.csv, these are in the order of the unseen    				   validation annotations so you can paste this directly into the unseen 			           version of validation_set.csv")



