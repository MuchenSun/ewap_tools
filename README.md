# Processing Tool for ETH Pedestrian Dataset

Original dataset available at: [https://icu.ee.ethz.ch/research/datsets.html](https://icu.ee.ethz.ch/research/datsets.html)

## How to use it?

[For now this tool does not support the Hotel dataset sequence yet.]

Step 0: Install dependencies

`python3 -m pip install opencv-python numpy matplotlib tqdm`

Step 1: Preprocessing

`cd ewap_tools/seq_eth/`

`mkdir frames`

`mkdir images`

`ffmpeg -i seq_eth.avi -vf fps=25 frames/frame%08d.png`

This process may take several minutes to finish, when it is done, go to the directory "ewap_tools/seq_eth/frames/", you should be able to see every frame of the original video as a png file.

Step 2: Run the example

`python3 example.py`

Feel free to check out the documented source code in `reader.py` to get more information. 

## Contact

Muchen Sun (muchen@u.northwestern.edu)
