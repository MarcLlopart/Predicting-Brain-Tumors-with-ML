# Pràctica Kaggle APC UAB 2021-22
### Nom: Marc Llopart Enajas
### Dataset: Brain Tumor from Jakesh Bohaju
### URL: [Kaggle](https://www.kaggle.com/jakeshbohaju/brain-tumor)

## Resum
El dataset utilitza estadístics sobre les ressònancies cerebrals. Conté 5 atributs de primer ordre i 8 de segon ordre que vindrien a ser estadístics sobre les textures de les imatges.

En total tenim 3762 files amb 13 atributs. Les dades que tenim són numèriques i volem classificar una variable categorica binària que ens indica si el pacient té o no un tumor al cap. Els atributs no venen normalitzats, no obstant els normalitzem per realitzar tot tipo de prova.

### Objectius del dataset
Amb el nostre dataset volem apendre a predir a partir dels estadístics de la seva ressònancia si persona té un tumor o no al cap.  
També serà interessant analitzar les imatges directament tractan-ho com un problema de visió.

## Experiments
En aquesta pràctica hem realitszat diversos experiments on hem començat netejant les dades i entenent-les. Per fer-ho hem visualitzat i buscat les correlacions entre els diversos atributs per seleccionar els més importants.  

Més endavant hem realitzat unes regressions lineals tot i que no tenen massa sentit ja que buscavem classificar una variable binària. De totes maneres algun atribut ens donava una bona precisió al classificar.

Tot seguit hem aplicat diversos models de Machine Learning com ara el Random Forest o varis SVM als quals els hi hem aplicat un *hyperparameter tuning* per trobar una aproximació als millors hiperparametres possibles, i així doncs obtenir les millors prediccions.

Un cop ja teniem els diversos models els hem entrenat i fet classificacions per veure el seu funcionament. Aquest funcionament l'hem avaluat a aprtir de les seves *accuracies*, les matrius de confusió i les diverses corbes de precisió com ho són la corba ROC i la Precision-Recall.

Per finalitzar la pràctica hem reservat un apartat a analitzar amb un model preentrenat de *pytorch* que ens ha ajudat a entrenar el nostre model per classificar les imatges directament.

### Model
#### Sense Hiperparàmetres
| Model  | Hieparàmetres |Mètrica|Temps|
| ------------- | ------------- |------------|-------------|
| Regressió Logística | Default |  98'8%          |0'023s |
| Decision Tree | Default |  98%          |0'013s |
| Random Forest | Default |  99%          |0'4s |
| Ada Boost Classifier | Default |  98'27%          |0'17s |
| XGBoost Classifier | Default |  98'4%          |0'32s |
| MLP | Default |  98'6%          |2'54s |

#### Amb Hiperparàmetres
| Model  | Hieparàmetres |Mètrica|Temps|
| ------------- | ------------- |------------|-------------|
| Regressió Logística | tol=0'1; solver='newton-cg'; penalty=none; dual=False; C=9.387816326530613 |  99'07%          |1'015s |
| Decision Tree | min_samples_split=10; min_samples_leaf=2; max_features='sqrt'; max_depth=35; cirterion='gini'  |  98'27%          |1'55s |
| Random Forest | n_estiamtors=612; min_samples_split=2; min_samples_leaf=2; max_features='log2'; max_depth=61; bootstrap=True|  99'33%          |1080'24s |
| Ada Boost Classifier | n_estimators=1566; learning_rate=0.2330909090909091; algorithm='SAMME' |  99'07%          |1031'55s |
| XGBoost Classifier | reg_lambda=0; reg_alpha=70; n_estimators=3093; max_depth=16; gamma=4; colsample_bytree=0'8 |  99'07%          |750'27s |
| MLP | solver='sgd'; learning_rate='adaptative'; hidden_layer_sizes=(50, 50, 50); alpha=0'0001; activation='tanh' |  98'93%          |232'31s |

#### Models SVM sense Hiperparàmetres
| Kernel  | Hieparàmetres |Mètrica|Temps|
| ------------- | ------------- |------------|-------------|
| Linear | Default |  97'27%          |0'036s |
| LinearSVC | Default |  97'17%          |0'0059s |
| RBF | Default |  97'44%          |0'036s |
| Polynomial | Default |  97'77%          |0'023s |
| Sigmoid | Default |  21'46%          |0'42s |

#### Models SVM amb Hiperparàmetres
| Kernel  | Hieparàmetres |Mètrica|Temps|
| ------------- | ------------- |------------|-------------|
| Linear | C=2.6633963963963967|  98'93%          |13'47s |
| LinearSVC | penalty='l2'; max_iter=54711; loss='hinge'; dual=True; C=4.865378378378379 |  98'8%          |3'30s |
| RBF | gamma='scale'; C=9.08918018018018 |  99'07%          |13'59s |
| Polynomial | gamma='scale'; C=0.3913513513513514; degree=4; coef0=0.23232323232323235 |  98'9%          |1785'54s |
| Sigmoid |  gamma='auto' ; C=9.329396396396396; coef0= 0.4040404040404041 |  98'8%          |78'54s |

#### Pretrained model
| Model  |Mètrica|Temps|
| ------------- | ------------- |------------|
| Squeezenet |  91'88%          |120 min 59s |

#### Models del Kaggle
| Model  |Mètrica|Temps|Link|
| ------------- | ------------- |------------|--- |
| MobileNet |  89%          |380s |[Kaggle](https://www.kaggle.com/angieashraf/89-brain-tumor-detection-using-dl)|
| Random Forest |  98'93%          |1s |[Kaggle](https://www.kaggle.com/angieashraf/89-brain-tumor-detection-using-dl)|

## Demo
Per veure com funcionen els nostres models podem fer-ho através del jupyter que trobarem al directori demo.

## Conclusions
El millor model creat ha estat el Random Forest desprès d'aplicar hiperparàmetres ja que a part de tenir la millor accuracy amb un 99'33% era el que millor classificava en general, i en especial els falsos negatius en els quals ens hem fixat per escollir.

En comparació amb els altres models vists al Kaggle obtenim una accuracy molt similar tot i que el temps varia sobretot en el cas del model preentrenat. Això pot ser a causa que en el kaggle utilitzaven les llibreries *keras* i *tensorflow* en comptes de *pytorch*.

## Idees per treballar en un futur

Seria interessant veure com millorar i optimitzar el temps al moment de classificar les imatges. Segurament ampliant el centre de la imatge que seria el cervell i retallant-la obtindriem alguna millora en el resultat.

## Llicència
El projecte s'ha desenvolupat sota llicència 
