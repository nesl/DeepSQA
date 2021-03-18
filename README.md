
# DeepSQA
**Codes for paper:**
[DeepSQA: Understanding Sensor Data via Question Answering(IoTDI 2021)](https://github.com/nesl/DeepSQA/blob/master/DeepSQA-paper.pdf)

Raw sensory data "OPPORTUNITY" can be found [here](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition)


### Environment:
- Python 3.7.7
- Nvidia RTX Titan
- Related packages: see "requirements.txt"
----


### Create folders to perform simulation:
```
mkdir sqa_data trained_models result source_dataset
```


## Files description:

-  __source_dataset:__ 
	- __opportunity:__ put raw sensory data "OPPORTUNITY" here.
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


- create folders: ``mkdir sqa_data trained_models result ``
-  __sqa_data:__ stores all the generated SQA data in json format, and aslo preprocessed data in .pkl and .npz format.
-  __trained_models:__ stores all the trained models. Models trained from a single simulation are stored in a single folder. e.g. "opp_sim1".
- __result:__ stores simulation result in .pkl form. e.g. "opp_sim1.pkl"

## Running experiments:
- __sqa_generation.py:__ scripts for generating the original SQA dataset. 
- __data_preprocessing.py:__ scripts for preprocess the original SQA dataset for training.
- __run_baselines&mac.py:__ codes for training either mac or baselines models.
- modify the parameters in the scripts for different simulation settings.



### Acknowledgement:

This research was sponsored by the U.S. Army Research Laboratory and the U.K. Ministry of Defence under Agreement # W911NF-16-3-0001. The views and conclusions contained in this document arethose of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the U.S. Army Research Laboratory, the U.S. Government, the U.K. Ministry of Defence or the U.K. Government. The U.S. and U.K. Governments are authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation hereon.

For more inforamtion, contact Tianwei Xing at:  [twxing@ucla.edu](mailto:twxing@ucla.edu)
