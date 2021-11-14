# Lunar Lander problem using a Deep Q-learning Neural Network
## 2021 Fall Hack-A-Roo Submission

![lander](https://user-images.githubusercontent.com/32967087/141695630-7c84e978-0fa8-43ce-b82c-148457dc5617.gif)

### Abstract<br>
This report briefly introduces the Lunar Lander Environment from OpenAI Gym. Solving this Lunar Lander using traditional Tabular Methods is practically challenging and expensive due to its large state space and complex reward structure. So, we turn to Function Approximations - An idea central to solving complex Reinforcement Leaning problems such as Lunar Lander through generalization of state space into a lower dimensional feature space. We specifically discuss NonLinear Function Approximation in detail. Then we discuss the role of Neural Networks as a Non-Linear Function Approximator and formulate a solution for Lunar Lander though a Deep Neural Network. Specifically, we will discuss the DQN algorithm introduced in (Mnih and et al 2015) paper and its application to solving Lunar Lander Environment using Deep Q-Network. DQN algorithm is know to be optimistic and it overestimates the action value function, to address this shortcoming we will briefly review the Double DQN Algorithm from (Hasselt, Guez, and Silver 2015) paper. Finally, a soft update optimisation is used for updating target network weights instead of a hard update proposed in (Mnih and et al 2015) paper for improving the learning performance on the Lunar Lander task and the results of this experiments are shared in detail including the hyper parameters used for training and the role of these hyper parameters in the learning process.
  
### Introduction<br>
OpenAI Gym is a toolkit for building and comparing Reinforcement Learning algorithms in a simulated environment. Lunar Lander is one such environment where the goal is to land a spaceship on the landing pad. Lunar Lander is a Continuous State Space Markov Decision Process(MDP) with states, actions and rewards described as below. States Lunar Lander has 8-dimensional state space vector, with six continuous states and two discrete states as below (x; y; ˙x; ˙y; θ;˙θ; legL; legR) Where state variables x and y are the current horizontal and vertical position of the lander, x˙ and y˙ are the horizontal and vertical speeds, θ and ˙θ are the angle and angular speed of the lander and legL and legR are the discrete binary values to indicate whether the left leg and right leg of the lunar lander are touching the landing pad. Actions It has four possible discrete actions - do nothing, fire the left orientation engine, fire the main engine, fire the right orientation engine to control the lander. Rewards For moving from top of the screen to the landing pad the lander has following rewards structure. If the lander moves away from the landing pad it is penalized the amount of reward that would be gained by moving towards the pad. An episode is finished if the lander crashes (or) comes to rest, receiving -100 or +100 points, respectively. Each leg-ground contact is worth +10 points. Firing the main engine incurs a cost of -0.3 points, firing the orientation engines incur a cost of -0.03 point for each occurrence. As we can infer from the above description of the Lunar Lander, it is a complex MDP with large state space and complex reward structures. For this reasons, Employing a Tabular Method to solve this MDP is practically hard by even
discretizing the continuous state space into bins, but its solvable with clever optimisations leading a complex solution. Instead, we will be exploring a much simpler solution to
solving the Lunar Lander Environment using Function Approximation techniques. Specifically we will be discussing Deep Neural Nets as function approximators to solving this MDP.

### Rewards
For moving from top of the screen to the landing pad the lander has following rewards structure. If the lander moves away from the landing pad it is penalized the amount of reward that would be gained by moving towards the pad. An episode is finished if the lander crashes (or) comes to rest, receiving -100 or +100 points, respectively. Each
leg-ground contact is worth +10 points. Firing the main engine incurs a cost of -0.3 points, firing the orientation engines incur a cost of -0.03 point for each occurrence.<br>

### State<br>
s[0] is the horizontal coordinate<br>
s[1] is the vertical coordinate<br>
s[2] is the horizontal speed<br>
s[3] is the vertical speed<br>
s[4] is the angle<br>
s[5] is the angular speed<br>
s[6] 1 if first leg has contact, else 0<br>
s[7] 1 if second leg has contact, else 0<br>

### Actions<br>
Four discrete actions available:<br><br>
0: do nothing<br><br>
1: fire left orientation engine<br><br>
2: fire main engine<br><br>
3: fire right orientation engine<br><br>

## How to use<br>
### Dependencies
* [Instructions for installing openAI gym environment in Windows](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30)

### Training model for lunar lander environment<br>
git clone https://github.com/srsapireddy/lunar_lander_using_DQN.git<br>
cd lunar_lander_using_DQN/<br>
Edit experiment parameters in train.py<br>
python3 train.py<br>
  
