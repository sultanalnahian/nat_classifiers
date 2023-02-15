# Naturalness Classifiers
Source code repository for Infoseeking and Relevancy classifiers which are used to measure the naturalness of open-ended conversational questions. In this code base I have trained and test two classifiers: Infoseeking and Relevancy. 

**Infoseeking Classifier:** It detects if a question is answerable/non-answerable given the context.

**Relevency Classifier:** It detects if the given question is relevant to the given context and conversation history.

## pre-requisite
python = 3.9, pytorch = 1.12.0, transformers = 4.14.1, cuda-toolkit=11.6, sentencepiece
## Creating Training Data 

The training data of both classifiers are generated automatically from QuAC dataset. To generate training data for the classifiers, take following steps  
1. Download the QuAC train and validation data from [here](https://quac.ai/)
2. Place the QuAC train and validation data file in a directory of the current project directory. The default location of the train file is source_data/QuAC/train_v0.2.json
3. Or the file location can be passed as argument.
4. Run data_gen_info_seeking.py file to generate training and validation data for Infoseeking classifier from QuAC data file. To generate training data pass QuAC train data file path in the --source_file argument. Similarly pass QuAC validation file path in --source_file argument to create validation file. Specifiy the output file path as well. Example:
```
python data_gen_info_seeking.py --source_file /QuAC/train_v0.2.json --output_file /data/train_infoseeking.json
```
5. Run data_gen_relevancy.py file to generate training and validation data for Relevancy classifier from QuAC data file same as step 4. Example:
```
python data_gen_relevancy.py --source_file /QuAC/train_v0.2.json --output_file /data/train_relevancy.json
```

## Training 

train.py contains the training script to train the models. To train infoseeking classifier, set the argument --task as "info-seeking". To train relevancy classifier, set it as "relevancy".

Example train command:
```
python train.py --task relevancy --train_data_file data/relevancy_train.json --validation_data_file data/relevancy_val.json
```
Other parameters we can set:

* dataloader_workers (type=int, default=2)
* epochs (type=int, default=15)
* learning_rate (type=float, default=3e-4) - For t5 models, recommended learning rates are 1e-4, 3e-4.
* max_length (type=int, default=512) - 512 is the maximum length for t5 models
* model (type=str, default="t5-small")
* save_dir (type=str, default="./models/t5-small") - Directory to save the trained model
* train_batch_size (type=int, default=16)
* valid_batch_size (type=int, default=16)
    
## Testing

We can test our model with the pre-trained models. Pre-trained models can be found in the following links:

### infoseeking pre-trained model
```
Please contact for the link
```
### relevancy pre-trained model
```
Please contact for the link
```
We can test our models by importing Eval class of inference.py file.



### Testing infoseeking model

This model takes concatenated question and context as input and predicts if the question is answerable/non-answerable given the context. The input format is as following:

`<question> question text here <context> context text here`

Example code:

``` python
from inference import Eval

eval = Eval(pre-trained-model)
prediction = eval.predict("<question> what is the largest river basin in the world? <context> Amazonia is the largest river basin in the world, and its forest stretches from the Atlantic Ocean in the east to the tree line of the Andes in the west.")
```


### Testing relevancy model

This model takes concatenated question, context and previous quesiton history as input and predicts if the quesiton is relevant to the given context and previous question histoy. The input format is as following:

`<quesiton> question text heare <context> context text here <history> <q> previous question 1 <q> previous question 2`

### test.py

We can also run test.py to test a file. Pass the task name (infoseeking or relevancy), pretraned model path, input file path and output file path as argument parameter.

Example command:
```
python test.py --task relevancy --model /models/relevancy --input_file data/relevancy_val.json --output_file output/results.tsv
```

The command will write an output file containing the prediction results of the classifier. 
