## Predict Shape Of Unicode Chars

Predict the Vietnamese characters


**Parameters to generate dataset and run prediction**

-c (category): Specify processing type for Digit | English chars | Vietnamese chars | Vietnamese accents
Values: digit | en | vn | ac


-n (new):
Values: 0 | 1  
-n=0: Load saved model  
-n=1: Force to retrain model


### Generate dataset

```
cd scripts
python dataset_generator.py -c=ac -n=1
```


### Run prediction

```
python simple_nn_run_loop.py -c=ac -n=1
```

