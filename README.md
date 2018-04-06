# Replication Package

In this replication package we make available **every** single script needed to **fully** replicate the results obtained in our paper.

## Requirements

* Java 1.8
* Maven 
* Python 3

One you install everything, you can use the following command to install every Python package.

`pip install -r requirements.txt`

## Steps

At first, the projects used for the evaluation need to be locally cloned.

```
python generate_script.py projects.csv clone
chmod 777 get_projects.sh
./get_projects.sh
```

Then, you can generate the scripts used to perform the mutation testing and then execute it.

```
python generate_script.py projects.csv run
chmod 777 run_experiment.sh
./run_experiment.sh
```

Please be **aware** that the entire process will take several hours. It is convenient to run it on a powerful dedicated server.

At the end of the mutation experiment, you have to aggregate the values for the obtained scores with:

```
python calcolate_results.py
```

After that, the command 
```
python aggregate_source.py
```
will merge in an unique frame the several metrics computed with third party tools. Please note that we provide those metrics pre-calculated since the employed tool are research prototypes not yet published!

At the end, executing 
```
python classifier.py
```
you will train and evaluate the machine learning classifier.
Please note that also this step takes a considerable amount of time.

## Docker image
We provide also a Dockerfile that can be used to build an image to replicate the results. Please note that, since you will create a Ubuntu image, you have to specify `python3` instead of the usual `python` (with is the version 2.7 by default)
