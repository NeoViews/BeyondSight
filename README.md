# BeyondSight

---

## Notes on the dataset from the Challenge Advisor

There are 14 datasets, each with 10000 rows and around 60-70 columns.

Each row corresponds to a video frame (there are usually 25 frames per second so a 90-minute video will have around 135000 frames).

The columns are for the XY coordinates of the players, the ball and the four corners of the camera views (you don’t have to use the and the camera corners).

There are usually 22 main players (11 each side) and there might be some substitutes so the total number of players might be between 22 and 28 (if 3 is the maximum number of substitutes)

Each player will have two columns for the X and Y coordinate. These columns might contain numeric values (when appearing in the camera view) or NaN if disappeared from the view. Each player might appear/reappear multiple times.

The ball has 2 columns of coordinates XY. The four corners of the camera will have 4x2=8 columns of XY data. These 2+8=10 columns will be the last 10 columns. The other columns are for the players (so if there are 62 columns, then 52 of these are for the players, i.e., there are 52/2=26 players in total)
