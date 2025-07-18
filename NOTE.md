# Note for different parts

### Note for augmented dataset
The data augmentation have the crop and resize for the image and track. The original track was left-down based coordinates, but everything in scenario map is left-top based coordinates, so I convert all the tracks to the left-top coordinates with same scale of the converted scenmap. For example, by default the scenario map is 10x than the orignial grid, so the tracks coordinates are also 10x bigger.

There is onething to NOTE is that the original dataset without scenmap still have the left-down and no scaled coordinates, but the original datset with scenmap have the left-top and scaled coordinates.


### Question: how to batch the augmented dataset?
