CONTENTS OF THIS FILE
---------------------

 * Introduction
 * Requirements
 * Recommended modules
 * Installation
 * Configuration
 * Troubleshooting
 * FAQ
 * Maintainers
----------------------

* Introduction:
    This project deals with a flask back-end server that runs a Thyroid Testing website. The main file is server.py file. The server.py 
    works as a gateway to the website sub-module. The website submodule deals with requests from user. It uses the Model Submodule to 
    make predictions. The 'Model' submodule uses a random forest Supervised Machine learning model loaded inside 'thyroid.sav'.

* Requirements:
    The following Packages are required for proper functioning:
        - click==8.0.3
        - colorama==0.4.4
        - feedparser==6.0.8
        - Flask==2.0.2
        - itsdangerous==2.0.1
        - Jinja2==3.0.3
        - joblib==1.1.0
        - MarkupSafe==2.0.1
        - numpy==1.21.4
        - scikit-learn==1.0.1
        - scipy==1.7.3
        - sgmllib3k==1.0.0
        - sklearn==0.0
        - threadpoolctl==3.0.0
        - Werkzeug==2.0.2
* Recommended:
    All the modules are necessary for proper functioning of the website and server.

* Configuration:
    The Host value inside $views.py$ can be changed to cater to the free port and ip in the user end.
    The HTML file shouldn't be altered much since it can affect the way input is being recieved by flask.

* Troubleshooting:
    The known issues that was faced while running the project.
        - The port specified is not free: This can be overcome by checking if the port is being free in Comman Prompt.
            - If the flask server is running already, check if the server.py has an existing instance and quit it.
        
        - Environment issue or Module not found issue.
            - Run   pip install -r requirements.txt.
                - This is caused because of missing modules mentioned in the Requirements.

* FAQ:
    - How to Run the program?
        - Navigate to .\env\Scripts\activate
        - python server.py or flask run.
        - Both commands work, however flask run is convenient.
    
    - How to get the required packages installed?
        - Run the following command -  pip install -r requirements.txt

    - What are the metrics used to distinguish between types  of Thyroid Disorder
        - There were 4 types of Thyroid Disorder based on NCBI (reference: https://www.ncbi.nlm.nih.gov/books/NBK285870/table/ch1.t1/) 
            - Overt Hyperthyroidism:
                - TSH<0.1 MlU/L or undetectable & Elevated T4 or T3

            - Overt Hypothyroidism:
                - TSH > 4.5 mlU/L & low T4

            - Subclinical Hyperthyroidism
                - TSH <  0.1 mlU/L & Normal T4 and T3
                - 0.1 to 0.4 mlU/L & Normal T4 and T3

            - Subclinical Hypothyroidism
                - TSH 4.5 to 10 mlU/L Normal T4
                - TSH >= 10 mlU/L Normal T4

    - What is the normal T3 value observed?    
        - NORMAL T3 = 0.9 to 2.8
* Maintainers:
    - Flask Backend - Aanand Dhandapani
    - Front End - Meghana Ramesh
    - Random Forest Model - Mrudula Krishna Prasad
