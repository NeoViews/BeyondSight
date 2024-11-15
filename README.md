<!-- HEADER -->
<div align="center">
  <!-- Source: https://tenor.com/view/football-soccer-ball-goal-golazo-gif-21897044 -->
  <img src="https://github.com/user-attachments/assets/d03a49d5-8eb5-440b-bc9d-75f4eabffde4" height=80>
  <h1>
    Beyond Sight
    <br><br>
  </h1>
</div>

<!-- DESCRIPTION -->
<img align=right src="https://github.com/user-attachments/assets/04f8bc7a-f4f5-44e5-9215-dd0752474c62" height=300>

<div align="center">
  This repository contains a machine learning-based dashboard designed to predict and visualize “invisible” players on a football field in real-time by analyzing video data frame by frame.
  <br><br><br>
  <strong>Contributors</strong>
  <br>
  Dakota Chang, Margaret Hollis, Paula Sefia, Toushar Koushik, and Wendy Liang
  <br><br><br>
  <strong>Timeframe</strong>
  <br>
  August 2024 - November 2024
</div>

<br><br><br><br>

<!-- TABLE OF CONTENTS -->
<h2>Table of Contents</h2>
<ol>
  <li><a href="#business-context">Business Context</a></li>
  <li><a href="#goals">Goals</a></li>
  <li><a href="#data-explanation">Data Explanation</a></li>
  <li><a href="#current-machine-learning-model-explanation">Current Machine Learning Model Explanation</a></li>
  <li><a href="#relevant-tools-and-libraries">Relevant Tools and Libraries</a></li>
  <li><a href="#generating-dashboard">Generating Dashboard</a></li>
  <li><a href="#next-steps">Next Steps</a></li>
  <li><a href="#license">License</a></li>
  <li><a href="#credits-and-acknowledgements">Credits and Acknowledgments</a></li>
</ol>

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>

<br><br>


<!-- BUSINESS CONTEXT -->
## Business Context

This project is based on sports data analytics, and we are leveraging raw video tracking data and machine learning tools/practices to deliver real-time analysis of football players’ positions.
<br>
<h3 align=center>Definitions</h3>
<ol>
    <li><strong>Football:</strong> The video data being analyzed will be a 90-minute recording of the game also referred to as soccer in the United States</li>
    <br>
    <li><strong>"Invisible" Players:</strong> Players who get substituted, players who are standing behind other players with respect to the camera’s perspective, or players who are cut out of the frame of the camera</li>
    <br>
    <li><strong>XY Positions:</strong> Within the training and testing datasets, the X and Y positions for each player for each frame of the video data is a pair of 2 columns per row (frame)</li>
    <br>
    <li><strong>CRISP-DM:</strong> Cross Industry Process for Data Mining, 6 phase methodology on how to transform a business problem into a data science based solution using measurable analytics</li>
  </ol>

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>

<br><br>

<!-- GOALS -->
## Goals

<img align=left height=200 src="https://github.com/user-attachments/assets/a425b070-0f1e-4b8d-83c5-54a7b5a39aad">

<p align=center> With a dataset of the XY Positions of players on the field every frame, our goal was to predict the positions of the players that either the cameras could not capture due to the camera’s angle and frame, substitute players, or other players not located on the field.</p>

<br>
  
<p align=center> By using Machine Learning models that can be trained and validated by our datasets, we planned to predict these “invisible” players with as much accuracy as possible.</p>

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>

<br><br>

<!-- DATA EXPLANATION -->
## Data Explanation

<img align=right src="https://github.com/user-attachments/assets/0a81614b-f00d-4981-98a8-fd69c32de85e">

<h3>How to Visualize Data in Simulation</h3>

In `Tracklets_visualisation.py` you can view the data in the form of a simulation. It is currently set to display collected data of on-screen players. You can adjust this (around line 230) to instead display `predicted.csv`, which (along with the other 13 datasets) are found in `BeyondSight/data/`

<h3>Training and Test Sets</h3>

There are 14 datasets in the form of CSV files, 13 of which are snippets of the 90-minute football game (our training sets) and 1 of which is the full 90-minute game (the testing set).

<h3>Rows and Columns</h3>

<strong>Rows: </strong> Each frame of the game is represented by a singular row
<br>
25 frames per second x a 90-minute video = ~135000 frames (rows)
<br><br>
<strong>Columns: </strong> The XY positions of each player, the ball, and the 4 corners of the camera views (or NaN if the positions aren't applicable)
<br>
25 players (2 x 11 main players and 3 substitutes) + 4 cameras + 1 ball = 60 pairs of XY coordinates

<h3>Notes</h3>
<ol>
  <li>Each player might appear/reappear multiple times</li>
  <li>The ball has 2 columns of coordinates XY which are the first two columns of the data</li>
  <li>The four corners of the camera will have 4x2=8 columns of XY data, which are the last 8 columns of each dataset</li>
  <li>The first 24 columns (12 players) are the away team (the 12th is the goalkeeper) and the latter 28 (14 players) are the home team (the 14th is the goalkeeper)</li>
  <li>There might be some substitutes so the total number of players might be between 22 and 28 (if 3 is the maximum number of substitutes), depending on the video</li>
</ol>

<h3>Goal</h3>
Use Machine Learning models to predict the XY positions that were not applicable (i.e. the positions that were labeled as NaN in the training set but found in the testing set)

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>

<!-- CURRENT MACHINE LEARNING MODEL EXPLANATION -->
## Current Machine Learning Model Explanation

<img align=left height=225 src="https://github.com/user-attachments/assets/46cfb14f-da29-4660-bcb6-9b08baaf5f6f">


After experimenting with the prospects of using different models we have researched, we decided to use a Long Short-Term Memory (LSTM) Model.
<br><br>
If you'd like to learn more about LSTM models, please visit [this website](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/)
<br><br>
With this model, we were able to compare multiple position-based variables to determine, using timestamps, the relative proximity of the invisible players with respect to the visible players
<br><br>
As seen in the dashboard, determining the overlap of the lines is how we were able to predict our accuracy



<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>

<!-- RELEVANT TOOLS AND LIBRARIES -->
## Relevant Tools and Libraries

See `requirements.txt` for more information.

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>

<!-- GENERATING DASHBOARD -->
## Generating Dashboard

After installing all the libraries in `requirements.txt`, run `python dashboard.py`. It will take a second to run but when it does an http link will appear. Copy and paste that into your browser of choice to see the dashboard.

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>

<!-- NEXT STEPS -->
## Next Steps

<strong>Improve Edge Cases:</strong> We noticed that there were certain trends that we did not account for, like penalty kicks, corner kicks, rapid player role switches, and goal celebrations, where players tend to gravitate towards certain areas habitually based on the nature of the game. We hope to find video data with these sorts of examples to learn how to fine tune our model to learn about the location of the "invisible" players in these scenarios without overfitting our model.
<br><br>
<strong>Advanced Analytics:</strong> One situation we tended to have hiccups with throughout this project was determining which columns were attributed to which players. We'd love to improve the simulation to have more clear depictions on the columns attributed to which player, camera, or ball, either with more clear documentation or shown in the simulation. Additionally, we'd love to change the simulation to include our projected locations of the invisible players, so we could see how close we were with respect to the simulation playing the entire game (the simulation processing the test set).


<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>


<!-- CREDITS AND ACKNOWLEDGEMENTS -->
## Credits and Acknowledgements

<img align="right" height=250 src="https://github.com/user-attachments/assets/8ea196db-79ed-4c9b-9dc0-99145e86c6c8">

<div align="center">
  Professor Tri-Dung Nguyen, our Challenge Advisor, for such a fascinating challenge project
  <br><br>
  NeoViews, a London-based sports data analytics startup, for the resources provided to start this project
  <br><br>
  Kailey Bridgeman, our Teaching Assistant, for her technical assistance
  <br><br>
  Everyone in the Break Through Tech Program who helped make the process of creating this project a success!
</div>

<p align="right">(<a href="https://github.com/NeoViews/BeyondSight/tree/main?tab=readme-ov-file#----beyond-sight------">back to top</a>)</p>
<br><br>
