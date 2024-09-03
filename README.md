# Tensor-Flow-Project--8--

<table>
  
**In this project we'll be applying Transfer Learning to solve very famous Food Vision 101 challenge of Image Classification using 100% percent i.e whole Food 101 dataset** <br></br>

**Let's looks into dataset:** <br></br>

**About Dataset**: <br></br>

You'll need train.csv, test.csv and sample_submission.csv.

**What should I expect the data format to be**?<br></br>

Each sample in the train and test set has the following information:<br></br>

1. The text of a tweet<br>
2. A keyword from that tweet (although this may be blank!)<br>
3. The location the tweet was sent from (may also be blank<br>

What am I predicting?<br></br>
You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.<br>

**Columns**: <br></br>
1. id - a unique identifier for each tweet<br>
2. text - the text of the tweet<br>
3. location - the location the tweet was sent from (may be blank)<br>
4. keyword - a particular keyword from the tweet (may be blank)<br>
5. target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)<br></br>

**Before jumping to the code lets understand what is TensorFlow and What do you mean by Transfer Learning, What is Classification problem and What do we actually understand by Feature Extraction and Fine Tuned Transfer Learning?**...<br></br>

**What is a TensorFlow?** <br></br>

TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. It was developed by the Google Brain team for Google's internal use in research and production. <br></br>

**What is a Transfer Learning?** <br></br>

Transfer learning is a machine learning technique in which knowledge gained through one task or dataset is used to improve model performance on another related task and/or different dataset. <br></br>

**What is a Classification?** <br></br>

Classification is a supervised machine learning process of categorizing a given set of input data into classes based on one or more variables. <br></br>

**What is a Feature Extractor Transfer Learning?** <br></br>
It Uses the representations learned by a previous network to extract meaningful features from new samples.<br></br>

**What is a Fine Tuned Transfer Learning?** <br></br>
Fine-tuning is a type of transfer learning. It involves taking a pre-trained model, which has been trained on a large dataset for a general task such as image recognition or natural language understanding and making minor adjustments to its internal parameters.<br></br>


**Important Note: Before Jumping to the code go through the Food Vision 101 dataset by clicking on this link (https://www.tensorflow.org/datasets/catalog/food101) to learn more. To understand more about these transfer learning methods type please visit this link to know them in detail (https://medium.com/munchy-bytes/transfer-learning-and-fine-tuning-363b3f33655d#:~:text=Fine%2Dtuning%20is%20a%20type,adjustments%20to%20its%20internal%20parameters.) For more information about transfer learning models used here go to (https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2) for ResNet50V2 to learn more about that and go to (https://arxiv.org/abs/2104.00298) to know about EfficientNet-V2.**

</table>

**So what are you waiting for...? Jump to the code to get started. As usual for any doubt or query see you in pull request section 😁😂. Thanks!**


