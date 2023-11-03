# Pipeline

There are several crucial details about the pipeline of MixMatch that aren't 
mentioned in the paper that makes or breaks the performance of the model. This
document aims to explain the pipeline in detail.

## Shortforms

- X: input data, in this case, images
- Y: labels of the input data
- K: number of augmentations, or to refer to the kth augmentation
- Lab.: labeled data
- Unl.: unlabeled data

## Data Preparation

The data is split into 3 + K sets, where K is the number of augmentations.

```mermaid
graph LR
    A[Data] --Shuffle--> B[Lab.]
    A --Shuffle--> C[Unl.]
    A --> D[Validation]
    A --> E[Test]
    B --Augment--> F[Lab. Augmented]
    C --Augment--> G[Unl. Augmented 1]
    C --Augment--> H[Unl. Augmented ...]
    C --Augment--> I[Unl. Augmented K]
```

See [Data Preperation](Data-Preparation.md) for more details.

## Model Architecture

We used a Wide ResNet 28-2 as the base model. This is a custom implementation
based on [YU1ut's PyTorch Implementation](https://github.com/YU1ut/MixMatch-pytorch/tree/master)
and [Google Research's TensorFlow Implementation](https://github.com/google-research/mixmatch)


## Training

Training is rather complex. The key steps are illustrated below.

To highlight certain steps, we use the following notation:

```mermaid
flowchart TD
    DDD[DATA]
    DDL[[DATA LIST]]
    PRC([PROCESS])
```

This is the pipeline of the training process.

```mermaid
flowchart TD
    
    YL[Y Lab.] --OneHot--> YLO[Y Lab. OHE]
    XUK[[X Unl. K]] --Predict--> YUPK[[Y Unl. K Pred.]]
    
    YUPK --> YUPA([Average Across K])
    YUPA --> YUP[Y Unl. Pred. Ave.]
    YUP --> YUPAS([Sharpen])
    YUPAS --> YUPR([Repeat K])
    YUPR --> YUPRK[[Y Unl. K Pred.]]
    
    XL[X Lab.] --> XC([Concat])
    XUK  --> XC
    YLO --> YC([Concat])
    YUPRK --> YC([Concat])
    
    XC --> X[X Concat.]
    YC --> Y[Y Concat.]

    X[Y] --> S([Shuffler])
    Y[X] --> S
    S --> XS[X Shuffled]
    S --> YS[Y Shuffled]
    X & XS --> MU([Mix Up])
    Y & YS --> MU
    MU --> XM[X Mix]
    MU --> YM[Y Mix]
    
    XM --> INT([Interleave])
    INT --Predict--> YMP[Y Mix Pred.]
    YMP --> RINT([Reverse Interleave])
    YM --> YMU[Y Mix Unl.]
    YM --> YML[Y Mix Lab.]
    RINT --> YMPU[Y Mix Pred. Unl.]
    RINT --> YMPL[Y Mix Pred. Lab.]
    
    YMU --> UL([Cross Entropy Loss])
    YMPU --> UL
    
    UL --> ULV[Unl. Loss Value]
    YML --> LL[Mean Squared Error]
    YMPL --> LL
    LL --> LLV[Lab. Loss Value]
    
    ULV --Unl Loss Scaler--> LS([Sum])
    LLV --> LS
    
    LS --> L[Loss]
    L --Backward--> M[Model]
```

We have both **Data** and **Data List**, as the augmentations create a new
axis in the data.

A few things to note:
- `Concat` is on the Batch axis, the 1st axis.
- `Predict` uses the model's forward pass.
  - The Label Guessing Prediction, `Predict(X Unl. K)`, doesn't use gradient.
- The Mix Up Shuffling is on the Batch axis, which includes the augmentations.
  If the data is of shape (B, K, C, H, W), the shuffling happens on both B and 
  K.
- CIFAR10 (and most datasets) are not even, use `drop_last` on the
  DataLoader to avoid errors.
### Sharpening

This is a step to make the Unlabelled Predictions more confident. This is done
by raising the predictions to a power, then normalizing the predictions.

```tex
\begin{align}
\text{Sharpen}(Y) &= \frac{Y^{\frac{1}{\tau}}}{\sum_{i=1}^{C} Y_i^{\frac{1}{\tau}}}
\end{align}
```

A higher `tau` value will make the predictions more confident.

### Mix Up

Mix Up mixes the original data with a shuffled version of the data. This ratio
of this mix is determined by a random **modified** sample from a Beta 
distribution. The modified sample is the maximum of the sample and its
complement.

```tex
\begin{align}
\text{Mix Up}(X, X_{\text{Shuffled}}) 
&= \hat\lambda X + (1 - \hat\lambda) X_{\text{Shuffled}}\\
\hat\lambda &= \max(\lambda, 1 - \lambda)\\
\lambda &\sim \text{Beta}(\alpha, \alpha)\\
\end{align}
```

Notably, when we modify the sample, we're effectively always taking the 
**larger** value, making the original sample **more prevalent** during the mix.

### Unlabelled Loss Scaler

> The original paper refers to this as the `lambda_u` hyperparameter.

The unlabelled loss scaler is a scalar that scales the unlabelled loss. This
linearly increases from 0 to 100 over the course of training.

The implementation is simple, just take the current epoch and divide by the
total number of epochs.

### Interleaving

Interleaving is not a well-documented step in the paper. See our 
[Interleaving](Interleaving.md) document for more details.

## Evaluation

The evaluation is simple. We just take the accuracy of the model on the test
set.