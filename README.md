# Diagnosing-patient-chest-Xrays-Multiclass-Classification
This project attempts to diagnose patients with COVID-19, viral pneumonia, and bacterial pneumonia from images of chest X-rays. The goal is to develop a multiclass classifier that achieves good weighted categorization accuracy on a set of unseen examples.

The data includes 1127 chest xrays drawn from several different sources (of varying size and quality) and a set of labels indicating whether each patient was healthy or diagnosed with bacterial pneumonia, viral pneumonia, or COVID-19. A 70:30 split was used to test the classifiers on the holdout validation set. For dimensionality reduction, principal component analysis (PCA) was used. Non-linear embedding approaches (multidimensional scaling and Isomap) were attempted but yielded lower accuracies.

Several algorithms were applied: support-vector machine (SVM), logistic regression, k-nearest neighbors, MLP neural network, and
random forest. The algorithms with the best weighted accuracies on the validation set were
to be determined as the best classifiers. Bayesian optimization was performed to optimize the hyperparameters of the classifiers. The best weighted accuracy was achieved by the SVM classifier, achieving around 78% balanced accuracy. The random forest classifier using
100 estimators performed weak across many metrics. However, this could be due to the fact that a higher
number of estimators were not used.

With exception to the COVID samples, the data is balanced. To fix the imbalance due to the
small minority of COVID samples, the common oversampling technique SMOTE (Synthetic Minority Over-sampling Technique) was used to increase the number COVID samples; this was followed up by random
under sampling of the other classes, as recommended by the creators of the SMOTE. This improved weighted
categorization accuracy.

------------------------
### Dependencies

- imblearn
- pandas
- skopt


&nbsp;

------------------------

