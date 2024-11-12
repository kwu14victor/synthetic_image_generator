This is a repository for creating synthetic cell images and labels.<br />
The created files can be used to train segmentation models.<br />

Usage (check sample script for more details):<br />
Use the SynCellImage class to create the data object.<br />
Next, use the SynCell class to create a cell and use the add_cells function of the SynCellImage class to add it to the data.<br />
Then, calling the image and label attributes of the SynCellImage object will give the data to train segmentation models.<br />
Moreover, calling the cell_dict attribute will allow the user to see the attributes of individual SynCell objects in the data.<br />

Observations/Comments:<br />
This script can create multiple objects, but the image quality needs manual trial and error.<br />
For example, assigning too many cells to an area can lead to insufficient visual features to train the segmentation model while the label still being presented.<br />
Therefore, it is critical to check the output before any experiment.<br />

TODO/Future work:<br />
Cell morphology pattern: The current cell morphology, including intensity distribution, is very simple and does not resemble real cells well. More literature review is necessary to model these real-world patterns.<br />

Requirements:<br />
numpy==1.21.6<br />
scikit-image==0.16.2<br />
matplotlib==3.2.2<br />

References:<br />
Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.<br />
Van der Walt S, Sch"onberger, Johannes L, Nunez-Iglesias J, Boulogne, Franccois, Warner JD, Yager N, et al. scikit-image: image processing in Python. PeerJ. 2014;2:e453.<br />
J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
