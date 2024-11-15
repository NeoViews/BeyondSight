<div align="center">
  <!-- Source: https://tenor.com/view/football-soccer-ball-goal-golazo-gif-21897044 -->
  <img src="https://github.com/user-attachments/assets/d03a49d5-8eb5-440b-bc9d-75f4eabffde4" height=80>
  <h1>
    Beyond Sight
    <br><br> August 2024 - November 2024
  </h1>
</div>

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

<h2>Table of Contents</h2>
  <ol>
    <li><a href="#business-context">Business Context</a></li>
    <li><a href="#goals">Goals</a></li>
    <li><a href="#data-explanation">Data Explanation</a></li>
    <li><a href="#model-explanation">Model Explanation</a></li>
    <li><a href="#relevant-tools-and-libraries">Relevant Tools and Libraries</a></li>
    <li><a href="#generating-dashboard">Generating Dashboard</a></li>
    <li><a href="#next-steps">Next Steps</a></li>
    <li><a href="#credits-and-acknowledgements">Credits and Acknowledgments</a></li>
  </ol>

<br><br>

## Business Context

This project is based on sports data analytics, and we are leveraging raw video tracking data and machine learning tools/practices to deliver real-time analysis of football players’ positions.
<br>
<h3 align=center>Definitions</h3>
<ol>
    <li><strong>Football:</strong> The video data being analyzed will be a 90-minute recording of the game also referred to as soccer in the United States</li>
    <li><strong>"Invisible" Players:</strong> Players who get substituted, players who are standing behind other players with respect to the camera’s perspective, or players who are cut out of the frame of the camera</li>
    <li><strong>XY Positions:</strong> Within the training and testing datasets, the X and Y positions for each player for each frame of the video data is a pair of 2 columns per row (frame)</li>
    <li><strong>CRISP-DM:</strong> Cross Industry Process for Data Mining, 6 phase methodology on how to transform a business problem into a data science based solution using measurable analytics</li>
  </ol>

<br><br>

## Goals

<img align=left height=200 src="https://github.com/user-attachments/assets/a425b070-0f1e-4b8d-83c5-54a7b5a39aad">

<p align=center> With a dataset of the XY Positions of players on the field every frame, our goal was to predict the positions of the players that either the cameras could not capture due to the camera’s angle and frame, substitute players, or other players not located on the field.</p>

<br>
  
<p align=center> By using Machine Learning models that can be trained and validated by our datasets, we planned to predict these “invisible” players with as much accuracy as possible.</p>

<br><br>

## Data Explanation


















<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

## Notes on the Dataset

There are 14 datasets, each with 10000 rows and around 60-70 columns.

Each row corresponds to a video frame (there are usually 25 frames per second so a 90-minute video will have around 135000 frames).

The columns are for the XY coordinates of the players, the ball and the four corners of the camera views (you don’t have to use the and the camera corners).

There are usually 22 main players (11 each side) and there might be some substitutes so the total number of players might be between 22 and 28 (if 3 is the maximum number of substitutes)

Each player will have two columns for the X and Y coordinate. These columns might contain numeric values (when appearing in the camera view) or NaN if disappeared from the view. Each player might appear/reappear multiple times.

The ball has 2 columns of coordinates XY which are the first two columns of the data. The four corners of the camera will have 4x2=8 columns of XY data. These 8 columns will be the last 8 columns. The other columns are for the players. The first 24 columns (12 players) are the away team (the 12th is the goal keeper) and the latter 28 (14 players) are the home team (the 14th is the goal keeper).

## Generating Dashboard

After install all the libraries in `requirements.txt`, run `python dashboard.py`. It will take a second to run but when it does an http link will appear. Copy and paste that into your browser of choice to see the dashboard.

## Visualizing Data

In `Tracklets_visualisation.py` you can view the data. It is currently set to display collected data of on screen players. You can adjust this (around line 230) to instead display `predicted.csv`

