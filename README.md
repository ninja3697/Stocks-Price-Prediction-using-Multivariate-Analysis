# Sequence Learning: Predicting Stock Prices using Multivariate Analysis 

## I) Abstract
In this paper, we build multivariate analysis models to predict stock price movement on Carriage Services, Inc. stocks data. Stock prices depends on various factors and their complex dynamics which makes them a difficult problem in real world. The purpose of this paper is to analyses the capability of a neural network to solve this problem efficiently. Recurrent Neural Networks (RNN) has demonstrated its capability of addressing complex time series problems. We analyzed different multivariate models based on different RNN architectures like GRU and LSTM and compares them with their univariate models and also with each other. We have used a soft computing approach based on RNN and models has been developed to find the temporal dependencies and forecast stock values of a particular company from its past history of stocks. We propose a multivariate neural network architecture in predicting stock price, and compare and contrast the prediction error of these models with univariate LSTM model. From the results, we infer that multivariate prediction models easily outperform univariate prediction models when trained on the same data. We also show that multivariate prediction models are more efficient and faster to train and deploy in business environment.


## II) Introduction

Stock price prediction is one of the most important business problems that has attracted the interest of all the stakeholders. To improve the performance, reliability of forecasting and the complexity of algorithms used in the process of solving this problem. However, the methods we have found yet are either based on simple linear regression assumptions (like ARIMA) or do not make full use of the data available and only consider one factor while forecasting (non-linear univariate models like ARCH, TAR [1] and deep learning models). Some researchers have also tried a combination of ANN and fuzzy logic [2] to use human like reasoning for this problem. But the stocks prediction is still open. The stock prices are highly dynamic and have non-linear relationships and is dependent on many factors at the same time [3]. We try to solve this problem of stock market forecasting using multivariate analysis.

Since multivariate time series have more features than univariate time series, they are more informative than the later one, so it is almost always better to use multivariate model to forecast the trend of complex systems like stocks. We attempt to help the research community to better understand this question and tried to find an answer for it. Recurrent Neural Networks (RNN) and its extensions like GRU and LSTM has shown good performances in other sequential data like sound waves, time series variations and in natural language processing.

We have used different deep learning techniques, namely RNN, GRU and LSTM to model our problem. It is proven that deep learning algorithms have the ability to identify existing patterns in the data and exploiting them by using a soft learning process [4]. Unlike other statistical and machine learning algorithms, deep learning architectures are capable to find short term as well as long term dependencies in the data and give good predictions by finding these hidden relationships.

We have proposed a 3-level methodology of our work. First, we preprocess the data to make the data multidimensional and suitable for our network architectures. Next, we split the data into train and test sets and train our models on the training data. At the final step, we make predictions using the models trained in the previous step on test data and calculate and analyze various error matrices. This paper isorganized in four parts. Part (III) presents the theoretical background on the various architectures used, part (IV) shows the methodologyused to conduct this experiment, part (VII) contains the results we obtained and conclusions are drawn in part (VIII).

## III) Literature Review

The following architectures are used in this paper:
  
  1) Recurrent Neural Networks (RNN):
  RNN are a class of ANNs where the output from previous step
  are fed as input to the current step along with the normal input. In
  feed forward ANNs, all the inputs and outputs are independent of
  each other, but in cases like when it is required to predict the time
  series, the previous values are required and hence there is a need
  to remember the previous values.
  It is found out that RNN suffers from vanishing gradient problem
  [5]. As we propagate the error through the network, it has to go
  through the temporal loop – the hidden layers connected to
  themselves in time by the means of weights wreck. Because this
  weight is applied many-many times on top of itself, that causes
  the gradient to decline rapidly. As a result, weights of the layers
  on the very far left are updated much slower than the weights of
  the layers on the far right. This creates a domino effect because
  the weights of the far-left layers define the inputs to the far-right
  layers. Therefore, the whole training of the network suffers, and
  that is called the problem of the vanishing gradient.

  2) Long Short Term Memory (LSTM):
  LSTM is an RNN network proposed by Sepp Hoch Reiter and
  Jürgen Schmidhuber in 1997 [6] to solve the problem of
  vanishing gradient in RNNs. LSTM uses the following gates to
  solve the problem:
  
   - Forget Gate: If set to true, the cell forgets the information coming from previous layers.
   - Input Gate: Chooses which value from input is going to update the memory state.
   - Output Gate: Chooses what will be the cell output on the basis of input and memory of the cell.

  3) Gated Recurrent Unit (GRU):
  It is a variation of RNN introduced by Kyunghyun Cho et al [7]
  in 2014. It is like a LSTM unit without an output gate. It has
  fewer parameters than LSTM and have less complexity. GRU
  have shown better performance than LSTM on certain smaller
  datasets, but it is still weaker than LSTM overall. 

## IV) Methodology

- **Raw Data:**
    
    We used the historical stock prices of Carriage Services, Inc. stocks
    obtained from Yahoo finance [8]. It contains 5670 records of daily
    stock prices of the stocks from 09/08/1996 to 22/02/2019. Each record
    contains information of high, low, opening and closing value of stocks
    as well as the volume of the stock sold on that day.

- **Data Pre-processing:**
    
    First, we remove some redundant and noisy data, such as the records
    with volume 0 and the records that are identical to previous record. For
    unifying the data range, we applied Min-Max normalization and
    mapped the values to a range of 0 to 1.  

    This data was split into train, validation and test data. The training data
    contains records from 1 Jan 1997 to 31 Dec 2006, validation data
    contains records from 1 Jan 2007 to 31 Dec 2008 and test data contains
    records from 1 Jan 2009 to 31 Dec 2010.


- **Training Process:**
    
    We train data on three sequential deep learning architectures, RNN,
    GRU and LSTM for our research. RNN is a special type of neural
    network where connections are made in a directed circle between the
    computational units. RNN make use of the internal memory to learn
    from the arbitrary sequence, unlike the feed forward neural networks.
    Each unit in an RNN has an activation function and weight. The
    activation function is time varying and real valued. The weights are
    modifiable. GRU and LSTM are extensions of RNN architecture. Each
    network we have created uses 3 layers of the respective RNN cell and
    a dense layer of 1 cell at the end.
    
- **Testing and Error Calculation:**

    Each model has been tested on the test set and their Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R2-score are
    calculated.



### Model 1: Univariate-LSTM:

The model is trained only on the Close price series of the dataset we
obtained; thus, it is a univariate model. Different parameters of the ANN are as
follows:

 - Timesteps: 40 
 - Neurons in each Layer: 40 and 35
 - Learning Rate: 0.001
 - Batch Size: 64
 - Total Trainable Parameters: 17408

The training data is fed to this network and the model is trained for 250
epochs on the training data and validated by the validation data.

### Model 2: Multivariate-RNN:

The model is trained on the series of records containing High price (Highest Correlation with target), Volume (Lowest Correlation with
target) and Close price of the stock. Different parameters of this ANN are as follows:

 - Timesteps: 30
 - Neurons in each Layer: 50 and 45
 - Learning Rate: 0.001
 - Batch Size: 32
 - Total Trainable Parameters: 7087

The training data is fed to this network and the model is trained for 150 epochs on the training data and validated by the validation data.

### Model 3: Multivariate-GRU:

The model is trained on the series of records containing High price
(Highest Correlation with target), Volume (Lowest Correlation with
target) and Close price of the stock. Different parameters of this ANN are as follows:

  - Timesteps: 40
  - Neurons in each Layer: 40 and 35
  - Learning Rate: 0.0001
  - Batch Size: 64
  - Total Trainable Parameters: 13359

The training data is fed to this network and the model is trained for 150
epochs on the training data and validated by the validation data.

### Model 4: Multivariate-LSTM:

The model is trained on the series of records containing High price
(Highest Correlation with target), Volume (Lowest Correlation with
target) and Close price of the stock.Different parameters of this ANN are as follows:

  - Timesteps: 50
  - Neurons in each Layer: 40 and 35
  - Learning Rate: 0.001
  - Batch Size: 64
  - Total Trainable Parameters: 17800

The training data is fed to this network and the model is trained for 200
epochs on the training data and validated by the validation data.
  

## V) Tools and Technology Used

We used Python syntax for this project. As a framework we used
Keras, which is a high-level neural network API written in Python. But
Keras can’t work by itself, it needs a backend for low-level operations.
Thus, we installed a dedicated software library — Google’s TensorFlow.

For scientific computation, we installed Scipy. As a development environment we used the Anaconda Distribution and Jupyter Notebook.
We used Matplotlib for data visualization, Numpy for various array
operations and Pandas for data analysis.

## VI) Results

The experiments were done for four deep learning models we have
trained. The models are cross validated on a window size of 600 records
and 5 splits. The final results obtained are shown in Table 1 below.

** Table 1: Results of different models on test data **

| Model | Features Used | MSE | RMSE | R2-score |
|---|---|---|---|---|
| Univariate-LSTM | Close | 0.0004030796 | 0.0185444448 | 0.9113916110 |
| Multivariate-RNN | [High,Volume,Close] | 0.0002176880 | 0.0139925408 | 0.9423308750 |
| Multivariate-GRU | [High,Volume,Close] | 0.0002792562 | 0.0155916908 | 0.9353505164 |
| Multivariate-LSTM | [High,Volume,Close] | 0.0004895794 | 0.0179982514 | 0.9214646906 |


The results from the table revealed that multivariate analysis not only
improves the performance of the model significantly but also reduces
the complexity of model (Univariate-LSTM model has 17408 trainable
parameters whereas Multivariate-RNN model has only 7087 trainable
parameters) making multivariate analysis a more efficient tool for
stocks prediction.

We also observe that the Multivariate-GRU and Multivariate-LSTM
models do not improve performance as much as expected and possible
reasons for it are:
  - The dataset used may not have long dependencies.
  - More data is required for these models for training.

## VII) Conclusion and Future Scope

We have proposed a multivariate neural network approach to solve the
problem of stock prices. We conclude that the multivariate ANN
models clearly outperform the best univariate ANN model (Univariate
LSTM). We also conclude that multivariate models make better use of
the data given and improves both performance and efficiency of the
stock prediction task. We proposed a multivariate deep learning-based
approach for predicting the stock prices. The approach we suggested
can only be solidified after comparing it with other methods of stock
prediction. We encourage researchers to also find out the reason of the
underperformance of the GRU and LSTM models of multivariate
analysis.


## VIII) References

  [1] K. Soman, V. Sureshkumar, V. T. N. Pedamallu, S. A. Jami, N. C.
  Vasireddy and V. K. Menon, “Bulk price forecasting using spark over
  nse data set,” Springer, 2016, pp. 137–146. International Conference
  on Data Mining and Big Data.

  [2] C. S. Lin, H. A. Khan and C. C. Huang, 'Can the neuro fuzzy model
  predict stock indexes better than its rivals?', Proc. CIRJE, CIRJE-F-
  165, Aug, 2002.

  [3] Z.P. Zhang, G.Z. Liu, and Y.W. Yang, “Stock market trend
  prediction based on neural networks, multiresolution analysis and
  dynamical reconstruction,” pp.155-56, March 2000. IEEE/IAFE
  Conference on Computational Intelligence for Financial Engineering,
  Proceedings (CIFEr).
  
  [4] Yoshua Bengio, I. J. Goodfellow, and A. Courville, “Deep
  learning,” pp. 436–444, Nature, vol. 521, 2015.

  [5] Razvan Pascanu, Tomas Mikolov, Yoshua Bengio, “On the
  difficulty of training Recurrent Neural Networks”, arXiv:1211.5063.

  [6] S. Hochreiter and J. Schmidhuber (1997). "Long short-term
  memory". Neural Computation. 1735 – 1780.
  doi:10.1162/neco.1997.9.8.1735.

  [7] Kyunghyun Cho (2014). "Learning Phrase Representations using
  RNN Encoder-Decoder for Statistical Machine Translation".
  arXiv:1406.1078.

  [8] https://in.finance.yahoo.com/quote/CSV/history/


