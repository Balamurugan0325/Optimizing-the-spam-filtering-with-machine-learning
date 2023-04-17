# SMS Spam Classifier 

### Overview

A simple Machine learning and `NLP` based Web Application which classify the given messages as Spam or Ham(Not Spam) built using `Flask` and deployed on `Heroku`.

- To view the Deployed Application, click on the link given below : 
  **SMS Spam Classifier Web Application** : *[https://sms-spam-predictor-nlp.herokuapp.com/](https://sms-spam-predictor-nlp.herokuapp.com/)*
  
- To get the Code for Exploratory data analysis/visualisations, Text-Processing, different algorithms used and the model evaluation, click on the link mentioned below :
  **Link of jupyter notebook -** *[https://github.com/asthasharma98/Data-Science/blob/main/SMS-Spam-Classifier/SMS%20Spam%20Classifier%20.ipynb](https://github.com/asthasharma98/Data-Science/blob/main/SMS-Spam-Classifier/SMS%20Spam%20Classifier%20.ipynb)*
  
**A Demo of the Web App :**

![Spam Messages Detector](https://github.com/asthasharma98/SMS-Spam-Classifier-Deployment/blob/master/Readme_resources/Spam.gif)

 ### Technical Aspect
 
 This Project is mainly divided into two parts:
 
 1. Exploring the dataset, Preprocessing texts and traning the model using `Sklearn`.
 2. Building and hosting a `flask` web app on `Heroku`.


**About the repository Structure :**

- Project consist `app.py` script which is used to run the application and is engine of this app. contians API that gets input from the user and computes a predicted value based on the model.
- `prediction.py` contains code to build and train a Machine learning model.
- *templates* folder contains two files `main.html` and `result.html` which describe the structure of the app and the way this web application behaves. These files are connected with Python via Flask framework.  
- *static* folder contains file `style.css` which adds some styling and enhance the look of the application. 


### Installation

The Code is written in Python 3.8. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after cloning the repository:

```
pip install -r requirements.txt 
```

*To clone the repository*

```
git clone https://github.com/asthasharma98/SMS-Spam-Classifier-Deployment.git
```


### Run 

*To Run the Application*

```
python app.py
```


### Deployement on Heroku

Install Heroku CLI as this makes it easy to create and manage your Heroku apps directly from the terminal. 
You can download it from [here](https://devcenter.heroku.com/articles/heroku-cli).

next step would be to follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.


### Technologies used 

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Heroku](https://github.com/jalbertsr/logo-badge-images/blob/master/img/rsz_heroku.png?raw=true)](https://www.heroku.com/)

[![Flask](https://github.com/jalbertsr/logo-badge-images/blob/master/img/rsz_flask.png?raw=true)](http://flask.pocoo.org/)  


### Future work 

- Improve model performance.
- Try to collect more samples of text messages with the help of users input.




