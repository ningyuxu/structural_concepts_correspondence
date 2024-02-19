# Conceptual Correspondence for Cross-Lingual Generalization

This repository contains code for the Paper *Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization* (Findings of EMNLP 2023). 


## Data

### Universal Dependencies

The [Universal Dependencies (UD) v2.10](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758) Treebank is used for our experiments, and we follow the split of training, development and test set in it.



## Experiments

### Model

The pretrained mBERT model ([bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)) and LLaMA-7B model ([decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)) are used for our experiments.


### Identification of Structural Concepts Based on Prototypes

For all our experiments that derive structural concepts from LLMs through a linear transformation, we train the linear probe with a batch size of 8 and a max sequence length of 128 for 20 epochs, and validate it at the end of each epoch. We select the model performing the best on the development set. We use the Adam optimizer with β1 = 0.9, β2 = 0.999, and a weight decay of 1e-6. The learning rate is set to 1e-4.

### Measuring Alignability between Structural Concepts in Different Languages

We use RSA and [Procrustes analysis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.procrustes.html) to measure the alignability
between structural concepts in different languages.

### Learning to Align Conceptual Correspondence

Our meta-learning-based method follows the procedure described above to derive prototypes for each concept. Subsequently, the networks are trained for 50 epochs, with a maximum sequence length of 128. During meta-training, given m languages, each epoch consists of 50 * m training episodes. These episodes are constructed using N labeled sentences as the support set and 30 labeled sentences as the query set. The parameters of the network are optimized through the Adam optimizer, with β1 = 0.9, β2 = 0.999, and a weight decay of 1e-4. The learning rate is set to 5e-5. The hidden layer dropout probability is 0.33.


### Aligning conceptual correspondence during in-context learning

For meta-learning during in-context learning, our networks are trained for 100 epochs with each consists of m * 10 episodes, where m = 5 is the number of languages involved in training. The parameters of the network are optimized through the Adam optimizer, with β1 = 0.9, β2 = 0.999, and a weight decay of 1e-4. The learning rate is set to 5e-4. The hidden layer dropout probability is 0.33.

