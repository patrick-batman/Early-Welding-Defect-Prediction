# Weldright-Techfest
## INTRODUCTION
In this project, we are building algorithms using the parameters to predict materials' welding defects by applying ML models. The main aim is to help the Godrej Aerospace Team produce defect-free, high-precision spacecraft components. In the following content of the abstract, we present the methodologies used on the provided dataset with 827534 rows and 88 columns to build a helpful algorithm.

## PRIMARY ANALYSIS OF THE DATASET
The first step of the model creation was the cleaning and preprocessing of the dataset. After performing various techniques, a clean dataset was obtained from which the following inferences were made:
1. In the dataset,  columns 13 to 87 only contain NaN values, and the first row of the data only describes the individual columns. Thus, columns 13 to 87 and row 1 can be dropped from the data frame for future analysis.
2. The dataset contains data for 24 days between 22/08/2019 and 19/09/2019, and 8 different welders are operating as per the given data.
3. A look at the defects column of the dataset tells us that there are primarily 2 kinds of defects observed in the welds-’ Tungsten defect’ and ‘Porosity defect.’
![dataset imbalance](https://user-images.githubusercontent.com/98745676/208961867-42172af4-d137-4914-95d0-5204cec9142a.png)
4. The dataset is heavily imbalanced.
5. A total of 4 operations and 16 different projects were undertaken during the period. 
6. The data includes many welds that occur at nearly 0 Voltage.
7. Most of the defects occur in the month of September which according to our hypothesis, corresponds to the gradual wear and tear of machines.  
8. On analysis of the defects on the basis of time, we found that most of the defects occurred during the time period of 4 PM to 7 PM(refer to the heatmap below). This might be due to a number of reasons such as non optimum conditions for welding, machine fatigue during the time or poor employee performance during the time due to the passive attitude of the workers post-lunch. The company can look into it and dig deeper to find out the actual reason and in turn greatly cut down on the losses due to welding in these time slots.


## OUR APPROACH

### Data Preprocessing
Firstly, we carried out an exploratory analysis of the dataset to find the factors that are more likely to affect the weld quality amongst those given in the dataset. Columns of Production and Machine were dropped as they are the same throughout the sample. The data was cleaned and preprocessed.
Columns containing corrupted values and values that seemed to provide no additional information to the model were removed.
Since we only need to predict the defects in welding accurately, we encoded the data in the defects column using LabelEncoder with 0 and 1 labels corresponding to no defect and a defect found in the welding process, respectively. We have now reduced the given problem to a binary classification problem.

### Feature Engineering

The next step is the Creation of Features , that is, feature engineering.
A total of four different features were created:
1. **porosityF**: 
porosityF=Flow*Job Temp*Humidity
Porosity is linked to gas flow, job temperature and humidity according to the research papers, so creating a feature porosityF increases the likelihood of detection of porosity defect. 
2. **HeatInput**: According to the standard formula, heat input is calculated as the product of Voltage and Current divided by the Speed of welding. 
Heat Input(here)=Voltage *CurrentFlow
However, since speed is unknown, a modified version of the formula is used where flow has replaced speed  
3. **Power**: It can be calculated as the absolute value of product of Voltage and Current. 
Power=absolute(Voltage * Current)
4. **(weld) Speed**: 
(Weld)Speed=Distance(assumed Constant)Time Difference
Technically defined as the speed with which the welder performs the welds, here information about the length of each weld was not given so it was assumed that each weld length is the same.

Label Encoding is done when categorical features need to be fed to the model; here it was done for Employee Code, Operation Number and Production Number.
However it led to poorer performance of the model and hence was dropped in the end.


### Feature Selection
<img width="452" alt="Screenshot 2022-12-21 at 10 35 00 PM" src="https://user-images.githubusercontent.com/98745676/208962951-66527fec-76cd-4655-92dd-d634abeee820.png">
Features were selected based on the relative importance.


## RELEVANT TECHNIQUES AND PERFORMANCE COMPARISON
### Balancing the Classes
The problem statement is an example of rare class detection, so methods of oversampling were used to increase the minority class population density.
SMOTE(Synthetic Minority Oversampling Technique) was applied to improve the class imbalance.

### Making Predictions
Based on the nature of the data, we tried training it using various classifier models such as SGDClassifier, RandomForestClassifier, CatBoost, XGBclassifier, Artificial Neural Networks,Angle-based Outlier Detection, and K Nearest Neighbours. 
Amongst them Angle-based Outlier Detection  and Random Forest Classifier seemed to work well for the model providing the most accurate results while making predictions. 

### ABOD(Angle-base Outlier Detection)
The model’s f1 score satisfies the criteria of being  >0.8.
The weighted average f1 score of the model is pretty high and unrealistic as the data was highly imbalanced and the entries for no defects are way greater than those with defects, hence macro f1 score is also calculated as we need to treat both the classes equal , but since weighted average f1 score has been mentioned as a judging criterion,so it’s necessary to calculate.

<img width="517" alt="Screenshot 2022-12-21 at 10 36 59 PM" src="https://user-images.githubusercontent.com/98745676/208963352-ab25c262-df7e-41fc-96af-b98ef8ead13c.png">

However the analysis of it’s confusion matrix would tell us that the model performs extremely poorly when it comes to actual detection of defects.Hence we suggest Random Forest Classifier.

### Random Forest Classifier
This model had a weighted F1 score of 0.78-0.85, which satisfies the criteria.
However it does an excellent job of predicting the defects.

<img width="373" alt="Screenshot 2022-12-21 at 10 38 09 PM" src="https://user-images.githubusercontent.com/98745676/208963598-e055b130-7112-4f2a-8745-d092c49a85cb.png">

As we can see the model has successfully predicted **658** defects out of 826 defects in total. This gives it an accuracy of **79.66%** .

<img width="334" alt="Screenshot 2022-12-21 at 10 39 01 PM" src="https://user-images.githubusercontent.com/98745676/208963772-170b04e9-7655-44b7-ac2c-dbfdc2b204f3.png">

Here are the different F1 scores for the model.
Hence the model is extremely successful in finding out the defects.


## ROI(Return On Investment)

### Breakdown Of Costs
In a properly functioning welding operation, costs are typically broken down like this:
Labour: 85 % 
Filler metals: 6 % 
Raw materials: 4 %
Shielding gas: 3 %
Electric power: 2 %

When a missed or defective weld goes undetected, these costs escalate at every stage in the welding operation and beyond.**However the monetary damage won’t be limited to a single weld; suppose a defective weld is catastrophic and leads to failure of the aircraft or defence vehicle. In such a case the cost would easily amount to hundreds of millions if not billions of dollars. Lawsuits and alike would also follow.**

The company can see the following benefits if it chooses to go forward with our suggestions:

1. Labour:According to our calculations Employee Number 382617 is the least efficient; so much so that his defective work alone is more than that of the top three performers combined.
One replacing him with worker who is as efficient as average of top three performers the the company is expected to save 300-400 defectsThus reducing the chance of catastrophic failure.

2. We found that most defects occur in the time period of 4PM to 7PM. The company can use this insight to reduce defects during this time and this can save upto 60 defects over a day.

<img width="1074" alt="heatmap-time date:defect" src="https://user-images.githubusercontent.com/98745676/208964424-0550ba94-bf2a-4097-9060-1c7dae7a2917.png">

So summing up after these insights the company saves 60 defects a day and 300-400 defects on the basis of employees.
So over a month (60*30+300=2100) that is more than 50% of defects can be reduced.


## TCO(Total Cost of Ownership)

![Screen-Shot-2021-05-20-at-8 10 53-AM-768x412](https://user-images.githubusercontent.com/98745676/208964968-c1ac2848-64ee-472e-b5a0-de864a2d3477.png)

The Total Cost of Ownership (TCO) is often the financial metric that you use to estimate and compare ML costs.It is divided into 3 parts, Acquisition Cost, Ownership cost and post-Ownership cost. Since this is an ML model , it does not need a large group of people at every factory to look after, a single team of Data scientists can look after every branch’s data , so the management cost is pretty low. On top of that the AWS charges and acquiring charges are also gonna be low in comparison to the profits it will yield by predicting the defects.
The model was created using the sci-kit-learn library, which is open-source and free of cost. The rest of the prices are estimated using AWS infrastructure and third-party engineering for deployment support. Choosing the MLOps framework, the total cost would be $94,500.


## Conclusion
The model is able to predict defects about **60% of the time even before they have occured.**
The f1 score of the model is  0.89 . The model when applied to practical uses will help in significant reduction of errors and help save companies resources.
Besides that our model provides key insights into the nature of defects as well. Once these suggestions are implemented the company can reduce **50-60%** of the defects.Other than that the model can predict 60% of the defects so in total we will be able to **bring down defective welds significantly.**
