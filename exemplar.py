'''
11.12.2024 ver 0.0.0 by KLW
This script aims demonstrate how to create synthetic cell images to train segmentation models
The project is originated from the data scientist interview for St-Pierre Lab at BCM
'''
from main import *

canvas = SynCellImage(120,120) #create the synthetic data object
for _ in range(10): #add ten random cells
    canvas.add_cells(SynCell((np.random.randint(10,110), np.random.randint(40,110)),\
                                 np.random.randint(20000,60000), np.random.randint(10,30),\
                                 np.random.uniform(1,2), np.random.randint(0,359)))
#visualize the cell image
plt.figure()
plt.imshow(canvas.image)#, cmap='afmhot')
plt.colorbar()
plt.show()
#visualize the cell label
plt.figure()
plt.imshow(canvas.label, cmap='viridis')
plt.colorbar()
plt.show()


io.imsave('cellimage_original.png', canvas.image)
io.imsave('celllabel_original.png', canvas.label)

canvas.delete_cells(2)
canvas.describe()
canvas.cell_dict[0].describe()
