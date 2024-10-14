# Basketball Attempts and Made Shots Detector

This project detects basketball attempts and made shots using computer vision based on the YOLO algorithm. It tracks the ball, players, and hoop to determine shot attempts and whether a shot was made. The model was custom-trained on the following classes: `ball`, `made`, `person`, `rim`, and `shoot`. 

It works best with videos of a single person shooting the ball. Future improvements will include functionality for 1v1 games, with further development planned to extend its capabilities.

Here's a preview of the results:

- [Demo Video 1](output_vids/demo1.mp4)
- [Demo Video 2](output_vids/demo2.mp4)

### Prerequisites
- Python 3.x
- Required packages are listed in `requirements.txt`. Install them by running:
```bash
pip install -r requirements.txt
```

### Running the Detector
By default, the program processes the video located at input_vids/vid29.mp4. To use the detector, simply run the following command:

``` bash
py bballvision.py
```
To process a different video, update the video_path variable inside bballvision.py to point to your desired video file:
``` python
video_path = 'input_vids/your_video.mp4'
```
The processed video, including detected attempts and made shots, will be saved in the output_vids folder.