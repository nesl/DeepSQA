# DeepSQA
**Codes for paper:**
[DeepSQA: Understanding Sensor Data via Question Answering(IoTDI 2021)](https://github.com/TianweiXing/DL_for_sensing_complex_env)

Public Github repo for DeepSQA can be found [here](https://github.com/TianweiXing/DL_for_sensing_complex_env)

Raw sensory data "OPPORTUNITY" can be found [here](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)


### Environment:
- Python 3.7.7
- Nvidia RTX Titan
- Related packages: see "requirements.txt"
----


### Create folders to perform simulation:
```
mkdir sqa_data trained_models result iotdi_fig
```


## Files description:

-  __source_dataset:__ 
	- __opportunity:__ put raw sensory data "OPPORTUNITY" here.
	- __opportunity.ipynb:__ script for analyzing opportunity dataset.
-  __sqa_data_gen:__ 
	- __question_family.ipynb:__ specifies all the question family templates used in generation
	- __question_family.json:__ stores the question family info in a json file.
	- __data_extraction.py:__ functions for extracting/splitting source data; generating scene_list; and visualizing data.
	- __dataset_analysis.py__ class of sqa dataset, used for analyzing the statistics.
	- __function_catalog.py:__ atomic function catalog
	- __functional_program.py:__ function programs associated with all question families
	- __question_generation.py:__ question generation function----given a sence, generate all questions of different families.
	- __sqa_gen_engine.py:__ question generation machine. Main program
	- __synonym_change.py:__ change words in generated question to increase linguistic variations.
	- __train_opp_model-single.ipynb:__ trains DL models on opp dataset natively. Trained model used in Neural Symbolic method.
	- __word_idx__

-  __preprocess_data:__ 
	- create folders: ``mkdir embeddings glove``
	- __embeddings:__ folder storing embedding matrix and word index
	- __glove:__ folder storing pretrained glove pre-trained word vectors. Downloaded from [here](https://nlp.stanford.edu/projects/glove/). (glove.6B.zip)
	- __embedding.py:__  create embedding matrix
	- __prepare_data.py:__ get sensory, question, and answer data in matrix form
	- __preprocessing.py:__ main function for converting a SQA dataset in json into processed .npz format.
	
-  __sqa_models:__ 
	- __mac_model:__ codes for DeepSQA-CA model (mac)
	- __baselines.py:__ codes for all other baseline models (prior, prior_q, SAN, conv-lstm, etc)
	- __run_baselines.py:__ training and testing baseline models 
	- __run_mac.py__ training and testing mac models

- __result_analysis:__ 
	- __utils.py:__ utility function for getting confusion matrix
	- __analyze_result.py:__ class for analyzing generated .pkl result.


- create folders: ``mkdir sqa_data trained_models result iotdi_fig``
-  __sqa_data:__ stores all the generated SQA data in json format, and aslo preprocessed data in .pkl and .npz format.
-  __trained_models:__ stores all the trained models. Models trained from a single simulation are stored in a single folder. e.g. "opp_sim1".
- __result:__ stores simulation result in .pkl form. e.g. "opp_sim1.pkl"
- __iotdi_fig:__ stores the generated figures (by __fig_plot.ipynb__) for IoTDI paper

## Running experiments:
- __sqa_generation.ipynb:__ scripts for generating the original SQA dataset. 
- __data_preprocessing.ipynb:__ scripts for preprocess the original SQA dataset for training.
- __run_baselines_cuda0.py:__ codes for training either mac or baselines models.
- __analyze_result.ipynb:__ general observation on sqa model performance, and also performance on prime dataset
- __check_consistency.ipynb:__ checking answer consistency against linguisic variations
- __fig_plot.ipynb:__ plots all the figures in IoTDI paper.



## About SQA dataset generation:

#### Dataset saving format:
- JSON files for Questions.
    - question	
	- question_family_index	
	- question_index	
	- answer	
	- context_index	
	- context_source_file	
	- context_start_point	
	- split (train, val, test)
- The raw sensory scene is not saved here. (need to get extracted when using it)

#### SQA dataset generated from OPPORTUNITY:
##### To get the Opportunity dataset: 
- curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip
- unzip the file into dataset folder.

##### To use the generated SQA dataset: (staled) 
- Download: https://drive.google.com/file/d/1AD7lVzPTI1o7oucpVveYWQCzujsVd6c2/view?usp=sharing
- unzip the "sqa_all_1800_450.json.zip" in generated_sqa_data folder. 

##### Generated data summary: 
- Time window is 1800 (30Hz x 60s), stride = 450.
- 16 question families
- 41186718 questions generated (2172758 unique)
- Question length: 5, 17.49, 28 (min, avg, max)
- Unique scenes: 1428
- Average num of questions per scene : 28842.23
- Unique answers: 1664 (39 if ignore duration count)


#### SQA DataGen Engine:

**Input:**
- Sensory data. (of any form)
- Sensory annotation. (encoding, also need a label decoder)
- Parameters: Window length (60s in Opportunity), stride length.

**Output:**
(Dump all data into a single json file)
- Questions json file.
	- Information of the generated data.
	- A list containing all questions.

##### **Inside the Engine** 

A file for all question families: "question_family.json"

Each question family includes:
- Multiple forms of possible texts.

For each question family:
- A functional program is associated with it. (composed of atomic functions in function catalog.)
- A python script for running the program and generating questions and answers in Natural Language form. 


----
#### Trained models can be accessed [here](https://drive.google.com/drive/folders/1CJLuGHvCbRasqrAG0O_jQaW5KrFwopdC?usp=sharing)

