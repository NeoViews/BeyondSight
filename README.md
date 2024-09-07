# BeyondSight

---

## Notes on the Dataset

There are 14 datasets, each with 10000 rows and around 60-70 columns.

Each row corresponds to a video frame (there are usually 25 frames per second so a 90-minute video will have around 135000 frames).

The columns are for the XY coordinates of the players, the ball and the four corners of the camera views (you don’t have to use the and the camera corners).

There are usually 22 main players (11 each side) and there might be some substitutes so the total number of players might be between 22 and 28 (if 3 is the maximum number of substitutes)

Each player will have two columns for the X and Y coordinate. These columns might contain numeric values (when appearing in the camera view) or NaN if disappeared from the view. Each player might appear/reappear multiple times.

The ball has 2 columns of coordinates XY which are the first two columns of the data. The four corners of the camera will have 4x2=8 columns of XY data. These 8 columns will be the last 8 columns. The other columns are for the players. The first 24 columns (12 players) are the away team (the 12th is the goal keeper) and the latter 28 (14 players) are the home team (the 14th is the goal keeper).
