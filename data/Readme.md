## Dataset preparation
- Download videos, video annotation,  questions and answers, and object proposals accordingly from the [official website](http://clevrer.csail.mit.edu/#)
- Transform videos into ".png" frames with ffmpeg.
- Organize the data as shown below.
    ```
    clevrer
    ├── annotation_00000-01000
    │   ├── annotation_00000.json
    │   ├── annotation_00001.json
    │   └── ...
    ├── ...
    ├── image_00000-01000
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   └── ...
    ├── ...
    ├── questions
    │   ├── train.json
    │   ├── validation.json
    │   └── test.json
    ├── proposals
    │   ├── proposal_00000.json
    │   ├── proposal_00001.json
    │   └── ...
    ```

- Download the [object proposals](http://clevrer.csail.mit.edu/#) from the region proposal network and follow the `Step-by-step Training` in [DCL](https://github.com/zfchenUnique/DCL-Release) to get object concepts and trajectories.

- We also provide data for learning physics in [Google Drive](https://drive.google.com/drive/folders/1vWnZoQYTxpvvwigxj_qnMYNyuHMjFF6Z?usp=sharing). You can download them optionally and put them in the `./data/` folder.