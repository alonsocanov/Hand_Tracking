# Hand Tracking

Simple script for hand tracking using mediapipe library.

More trainig examples are needed and a validation dataset is needed.

In order to make the landmarks irrelevant to the positon in the image distances are clalculated bettween al llandmarks and a specific landmark, to make detection better two landmarks could be useded for distances. Then this distance is normaliced between two lanlandammrs that usually don chance.

## Images

- Image of the hand signal recognition
  ![alt text](https://github.com/alonsocanov/Hand_Tracking/blob/master/data/img_2.jpg "Image")
- Video feed
  ![alt text](https://github.com/alonsocanov/Hand_Tracking/blob/master/data/out_2.gif "Video")

## References

- Link for the [mediapipe library](https://google.github.io/mediapipe/solutions/hands.html)

- Link to a helpfull [video from Murtaz's Workshop](https://www.youtube.com/watch?v=NZde8Xt78Iw&t=2479s) on how to use the mdiapipe library
