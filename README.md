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

This needs to be a description of the model used and a brief overview of how it works in theory (e.g taken of a CNN Model): 


**Rectified Linear Unit (ReLU)** is the activation layer used in CNNs.The activation function is applied to increase non-linearity in the CNN. Images are made of different objects that are not linear to each other.


**Max Pooling:** A limitation of the feature map output of Convolutional Layers is that they record the precise position of features in the input. This means that small movements in the position of the feature in the input image will result in a different feature map. This can happen with re-cropping, rotation, shifting, and other minor changes to the input image. A common approach to addressing this problem from signal processing is called down sampling. This is where a lower resolution version of an input signal is created that still contains the large or important structural elements, without the fine detail that may not be as useful to the task.

## Future Work
Good ideas or strategies that you were not able to implement which you think can help  improve performance.
