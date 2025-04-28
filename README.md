# Problem Statement 
In the bustling world of Kanto, where Pokémon battles shape destinies, crime lurks in the shadows. Detective Kotso, the sharpest mind in Pokémon crime investigations, has been tasked with an urgent mission. **The mayor suspects that Team Rocket has infiltrated the city, disguising themselves as ordinary citizens.**

But Kotso doesn’t work alone—he relies on you, a brilliant data scientist, to uncover the truth. Your job? **Analyze the data of 5,000 residents to predict which of the 1,000 unclassified individuals are secretly part of Team Rocket.**

## Dataset

This dataset contains demographic, behavioral, and battle-related information of 5,000 Pokémon world citizens. It is the [Pokemon Detective: Unmask Team Rocket](https://www.kaggle.com/datasets/kotsop/pokmon-detective-challenge) from Kaggle.
The class labels are:

**1. ID**  
Unique identifier for each citizen.

**2. Age**  
Age of the citizen.

**3. City**  
City the citizen currently lives in.

**4. Economic Status**  
Socioeconomic class of the citizen.  
Categorical values: `Low`, `Middle`, or `High`.

**5. Occupation**  
Profession in the Pokémon world (e.g., Gym Leader, Researcher, Nurse, etc.).

**6. Most Frequent Pokémon Type**  
The Pokémon type most frequently used by the citizen in battles (e.g., Fire, Water, Psychic, etc.).

**7. Average Pokémon Level**  
Average level of the Pokémon owned by the citizen.

**8. Criminal Record**  
Indicates if the citizen has a criminal record.  
Values:  
- `0` = Clean  
- `1` = Dirty

**9. Pokéball Usage**  
The type of Pokéball most frequently used by the citizen (e.g., DarkBall, UltraBall, TimerBall).

**10. Winning Percentage**  
Win rate in Pokémon battles, represented as a percentage (e.g., 64%, 88%).

**11. Gym Badges**  
Number of official gym badges earned (range: 0 to 8).

**12. Is Pokémon Champion**  
Boolean indicator.  
`True` if the citizen has defeated the Pokémon League (Elite Four).

**13. Battle Strategy**  
Typical battle approach.  
Possible values: `Defensive`, `Aggressive`, `Unpredictable`.

**14. City Movement Frequency**  
Number of times the citizen changed cities in the past year.

**15. Possession of Rare Items**  
Boolean indicator of whether the citizen possesses rare or legendary items.  
Values: `Yes` or `No`.

**16. Debts to the Kanto System**  
Total amount of debt owed by the citizen to the Kanto government (e.g., 20,000).

**17. Charitable Activities**  
Boolean indicator.  
`Yes` if the citizen has participated in registered charity events.

**18. Team Rocket Membership**  
**Target variable.**  
Indicates whether the citizen is a secret member of **Team Rocket**.  
Values:  
- `Yes` = Member of Team Rocket  
- `No` = Not affiliated

## Model(s) Used

Two main models were implemented for Team Rocket member detection:

**1. Logistic Regression**
A linear classification model that estimates the probability of a binary outcome. It works by:
- Applying a linear combination of features
- Transforming the output through a sigmoid function to get probabilities between 0 and 1
- Using "saga" solver to handle both L1 and L2 regularization
- Setting max_iter=7000 to ensure convergence with our complex feature set

**2. Random Forest Classifier**
An ensemble learning method that operates by:
- Constructing multiple decision trees during training
- Having each tree "vote" on the final classification
- Combining these votes to make the final prediction
- Using feature randomness to create diverse trees and prevent overfitting

**Hyperparameter Optimization**
GridSearchCV was employed to fine-tune the Random Forest model by:
- Testing different combinations of parameters:
  - n_estimators: [10, 50, 100, 200]
  - max_depth: [None, 10, 20, 30]
  - min_samples_split: [2, 5, 10]
  - min_samples_leaf: [1, 2, 4]
  - bootstrap: [True, False]
- Using 5-fold cross-validation to ensure robust performance
- Selecting the best parameter combination based on validation scores

**Class Imbalance Handling**
Both models were tested with various balancing techniques:
- Class weights
- SMOTE (Synthetic Minority Over-sampling Technique)
- Combination of SMOTE and class weights

## Future Work

Several potential improvements could enhance the model's performance:

1. **Feature Engineering**
   - Create interaction features between related variables
   - Develop more sophisticated behavioral patterns analysis
   - Include temporal analysis of citizen activities

2. **Model Enhancements**
   - Implement stacking with other classifiers (XGBoost, LightGBM)
   - Explore deep learning approaches for complex pattern recognition
   - Develop custom loss functions that penalize false negatives more heavily

3. **Deployment Considerations**
   - Develop real-time prediction capabilities
   - Implement model monitoring for concept drift
   - Create an interpretability dashboard for law enforcement use