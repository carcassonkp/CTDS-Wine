# [Wine Quality](http://www3.dsi.uminho.pt/pcortez/wine/)
Using the wine quality dataset do the classification, regression, clustering and dimensionality reduction.


<p align="center" width="100%">
    <img width="100%" src="https://user-images.githubusercontent.com/70576587/149422436-fe1100f2-e8a5-43d9-bf54-2b87677a352a.png"> 
</p>

## Classification Accuracies

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/70576587/149364232-b8a89079-cf71-40b9-8d34-4c11c604d771.png"> 
</p>

### Train set
Since the model was having difficulties predicting classes with a low number of samples, SMOTE was used with the Random Forest classification, but it's accuracy was worse than the normal Random Forest classification with the original train set.

Create a new training set using SMOTE(Synthetic Minority Over-Sampling).
The algorithm creates synthetic minority class samples to increase the representation of minority classes.

Before SMOTE
![newplot](https://user-images.githubusercontent.com/70576587/149509975-1ee7f439-514d-43a0-af01-31efa47bf03b.png)

After SMOTE 
![newplot(1)](https://user-images.githubusercontent.com/70576587/149510178-880e9211-9a0c-473d-a581-54f9fc468648.png)

## Regression

### R2 Score

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/70576587/149511454-0a1c2308-5232-4ff9-af19-c7ce9961d118.png"> 
</p>

### MSE

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/70576587/149512229-a5f06f02-af6d-4c3b-ba7e-a8009037fc5a.png"> 
</p>

## Clustering Graphs
![Clustering](https://user-images.githubusercontent.com/70576587/149420560-79f21de9-d266-41d7-91f6-22fc08f28648.png)
```
K means
  accuracy                           0.07      1599
Mean shift
  accuracy                           0.00      1599
Gaussian mixture
  accuracy                           0.20      1599
```
## Dimensionality Reduction

### PCA

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/70576587/149518888-63f82207-f68e-43e1-86d0-dfa15b6a5c57.png"> 
</p>
