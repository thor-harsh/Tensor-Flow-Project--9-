# Tensor-Flow-Project--9--

<table>
  
**In this project we'll be applying Natural Language Processing to solve Natural Language Processing with Disaster Tweets and clssifying them in different classes of tweets.** <br></br>

**Let's looks into dataset:** <br></br>

**About Dataset**: <br>

You'll need train.csv, test.csv and sample_submission.csv.<br></br>

**What should I expect the data format to be**?<br>

Each sample in the train and test set has the following information:<br></br>

1. The text of a tweet<br>
2. A keyword from that tweet (although this may be blank!)<br>
3. The location the tweet was sent from (may also be blank<br></br>

**What am I predicting**?<br></br>
You are predicting whether a given tweet is about a real disaster or not. If so, predict a 1. If not, predict a 0.<br></br>

**Columns**: <br>
1. id - a unique identifier for each tweet<br>
2. text - the text of the tweet<br>
3. location - the location the tweet was sent from (may be blank)<br>
4. keyword - a particular keyword from the tweet (may be blank)<br>
5. target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)<br></br>

**Before jumping to the code lets understand what is TensorFlow and What do you mean by Transfer Learning, What is Classification problem and What do we actually understand by Feature Extraction and Fine Tuned Transfer Learning?**...<br></br>

**What is a TensorFlow?** <br></br>

TensorFlow is a free and open-source software library for machine learning and artificial intelligence. It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks. It was developed by the Google Brain team for Google's internal use in research and production. <br></br>

**What is a Natural Language Processing?** <br># 1. Setup token inputs/model
token_inputs = layers.Input(shape=[], dtype=tf.string, name="token_input")
token_embeddings = tf_hub_embedding_layer(token_inputs)
token_output = layers.Dense(128, activation="relu")(token_embeddings)
token_model = tf.keras.Model(inputs=token_inputs,
                             outputs=token_output)

# 2. Setup char inputs/model
char_inputs = layers.Input(shape=(1,), dtype=tf.string, name="char_input")
char_vectors = char_vectorizer(char_inputs)
char_embeddings = char_embed(char_vectors)
char_bi_lstm = layers.Bidirectional(layers.LSTM(25))(char_embeddings) # bi-LSTM shown in Figure 1 of https://arxiv.org/pdf/1612.05251.pdf
char_model = tf.keras.Model(inputs=char_inputs,
                            outputs=char_bi_lstm)

# 3. Concatenate token and char inputs (create hybrid token embedding)
token_char_concat = layers.Concatenate(name="token_char_hybrid")([token_model.output, 
                                                                  char_model.output])

# 4. Create output layers - addition of dropout discussed in 4.2 of https://arxiv.org/pdf/1612.05251.pdf
combined_dropout = layers.Dropout(0.5)(token_char_concat)
combined_dense = layers.Dense(200, activation="relu")(combined_dropout) # slightly different to Figure 1 due to different shapes of token/char embedding layers
final_dropout = layers.Dropout(0.5)(combined_dense)
output_layer = layers.Dense(num_classes, activation="softmax")(final_dropout)

# 5. Construct model with char and token inputs
model_4 = tf.keras.Model(inputs=[token_model.input, char_model.input],
                         outputs=output_layer,
                         name="model_4_token_and_char_embeddings")

Natural language processing (NLP) is a subfield of computer science and artificial intelligence (AI) that uses machine learning to enable computers to understand and communicate with human language.  <br></br>

**What is a Classification?** <br></br>

Classification is a supervised machine learning process of categorizing a given set of input data into classes based on one or more variables. <br></br>

**Important Note: Before Jumping to the code go through the Natural Language Processing with Disaster Tweets dataset by clicking on this link(https://www.kaggle.com/competitions/nlp-getting-started/data).**

</table>

**So what are you waiting for...? Jump to the code to get started. As usual for any doubt or query see you in pull request section 😁😂. Thanks!**


