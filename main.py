'''
11.11.2024 ver 0.0.0 by KLW
This script aims to generate synthetic cell images to train segmentation models
The project is originated from the data scientist interview for St-Pierre Lab at BCM
'''
import numpy as np
from skimage import morphology, io, transform
import matplotlib.pyplot as plt

class SynCellImage:
    '''
    Cell image class where the user can add new cells into the image
    '''
    def __init__(self, height=256, width=256):
        '''
        Initialize the object and save the metadata of the image class
        Arguments:
        height (int): the height of the synthetic image
        width (int): the width of the synthetic image
    
        Returns a SynCellImage object with all the cells in the image, image, and label
        '''
        if isinstance(height, int) and isinstance(width, int):
            #initialize the attributes of the object
            self.height = height
            self.width = width
            self.image = np.zeros((self.height, self.width), dtype=np.uint16) #store the cell image
            self.label = np.zeros((self.height, self.width), dtype=np.uint8) #store the cell label
        else:
            raise ValueError("Both inputs must be integers.")
        #The following two attributes keep track of the cells added to the image
        self.cell_id = 0
        self.cell_dict = []

    def add_cells(self, cell):
        '''
        Function to add a cell to the synthetic cell image

        Arguments:
        cell (SynCell): the synthetic cell object to add to the data
        '''
        if self.cell_id>=255:
            #8-bit label image can only store 256 cells
            print('maximum number of cell reached')
            return
        if isinstance(cell, SynCell):
            self.cell_id+=1 #update cell id
            shape = list(cell.cell_img.shape)
            #crop the patch if it will exceed the field of view
            if shape[0]>self.height:
                shape[0] = self.height
            if shape[1]>self.width:
                shape[1] = self.width
            self.cell_dict.append(cell) #store the metadata of the cell added
            self.image[:shape[0], :shape[1]] =\
                np.where(cell.cell_mask[:shape[0], :shape[1]]==1,\
                cell.cell_img[:shape[0], :shape[1]], self.image[:shape[0], :shape[1]])
                #update the image
            self.label[:shape[0], :shape[1]] =\
                np.where(cell.cell_mask[:shape[0], :shape[1]]==1,\
                          self.cell_id, self.label[:shape[0], :shape[1]])
                #update the label
        else:
            raise ValueError("Please only use the SynCell class.")

    def delete_cells(self, cell_id):
        '''
        Delete selected cell from the image by referring to cell id
        '''
        if len(self.cell_dict)>cell_id-1 and self.cell_dict[cell_id]:
            old_cell = self.cell_dict[cell_id]
            shape = list(old_cell.cell_img.shape)
            if shape[0]>self.height:
                shape[0] = self.height
            if shape[1]>self.width:
                shape[1] = self.width
            #remove cells from the image and label by assigning them to zeroes
            self.image[:shape[0], :shape[1]] =\
                np.where(old_cell.cell_mask==1, 0, self.image[:shape[0], :shape[1]])
            self.label[:shape[0], :shape[1]] =\
                np.where(old_cell.cell_mask==1, 0, self.label[:shape[0], :shape[1]])
            self.cell_dict[cell_id] = None
        else:
            print('cell not found')
            #exit the loop if the id assigned is invalid
            return

class SynCell:
    '''
    Synthetic cell class
    I use gaussian pattern to assign the intensity distribution
    Via visual check, I picked the minimal valid intensity at 8500 (can be further modified)
    The cell will be initialized as circular before applying further modifications, if any
    '''
    def __init__(self, centroid, intensity, size, aspect_ratio = 1, rotation = 0, min_int=8500):
        '''
        Initialize the object and save the metadata of the image class
        After that, generate the cell image and mask, followed by intensity modification and padding
        
        Arguments:
        centroid (tuple of two integers): centroid of the cell when initialized as circular
        intensity (int): maximum intensity value of the cell
        size (int): the size, in radius, of the cell
        aspect_ratio (float): the geometry of the cell, defined as the formula for ellipses
        rotation (float): rotation of the cell
        min_int (int): minimal valid value for a pixel to be considered as a cell
    
        Returns a SynCell object with the metadata, image, and label
        '''
        centroid_x, centroid_y = centroid[0], centroid[1]
        #check if input format is valid
        assert isinstance(centroid_x, int), "Centroid coordinates must be integers."
        assert isinstance(centroid_y, int), "Centroid coordinates must be integers."
        assert 0<=intensity<=65535, "Please assign intensity value within 16-bit range."
        assert (0<size and isinstance(size, int)), "Cell size should be positive integer."
        assert isinstance(min_int, int), "Please assign minimum intensity as integer."
        assert aspect_ratio>0, "Please assign aspect ratio as positive integer."
        assert isinstance(rotation, (int, float)), "Please assign rotation angle as numeric value."
        #save the attributes
        self.centroid_x = centroid_x
        self.centroid_y = centroid_y
        self.intensity = intensity
        self.aspect_ratio = aspect_ratio
        self.size = size
        self.min_int = min_int
        self.rotation = rotation
        #generate cell as a circle with uniform intensity
        self.draw_cell_block()
        #modify intensity distribution
        self.modify_intensity()
        #pad the cell to map onto the image
        self.pad_cell()
        self.cell_img = self.cell_img.astype(np.uint16)

    def draw_cell_block(self):
        '''
        Initialize the cell as a circular object
        '''
        self.cell_mask = morphology.disk(self.size).astype(bool)
        self.cell_img = np.where(self.cell_mask==1, self.intensity, 0)

    def modify_intensity(self, mode = 'center'):
        '''
        Apply the Gaussian pattern to the intensitt distribution of the cell image
        '''
        self.cell_img = apply_gaussian(self.cell_img)
        self.cell_img = np.where(self.cell_mask==1, self.cell_img, 0)
        self.cell_mask = np.where(self.cell_img>self.min_int, 1, 0)
        if self.aspect_ratio!=1: #change the cell geometry if necessary
            self.cell_img = transform.resize(self.cell_img,\
                             (int(self.size*self.aspect_ratio), self.size))
        if self.rotation!=0: #rotate the cell if necessary
            self.cell_img = transform.rotate(self.cell_img, self.rotation)

        self.cell_mask = np.where(self.cell_img>self.min_int, 1, 0)
        if mode!='center':
            eroted_mask = morphology.binary_erosion(self.cell_mask, structure=morphology.disk(3))
            noise = np.random.random(self.cell_mask.shape) > 0.98
            self.cell_mask = np.logical_and(self.cell_mask, np.logical_and(eroted_mask, noise))

    def pad_cell(self):
        '''
        Align the image patch of this cell to the origin of the overall image by padding
        Use the assigned coordinates of the centroid can help us do the math 
        '''
        row_pad = self.centroid_x-self.size
        col_pad = self.centroid_y-self.size
        if row_pad<0:
            self.cell_mask = self.cell_mask[-row_pad:, :]
            self.cell_img = self.cell_img[-row_pad:, :].astype(np.float64)
            row_pad = 0
        if col_pad<0:
            self.cell_mask = self.cell_mask[:, -col_pad:]
            self.cell_img = self.cell_img[:, -col_pad:].astype(np.float64)
            col_pad = 0
        if col_pad>0 or row_pad>0:
            self.cell_mask = np.pad(self.cell_mask, ((row_pad, 0), (col_pad,0)),\
                              'constant', constant_values=(0, 0))
            self.cell_img = np.pad(self.cell_img, ((row_pad, 0), (col_pad,0)),\
                              'constant', constant_values=(0, 0))


def apply_gaussian(image, random = False, sigma=(7,7)):
    '''
    Compute a gaussian filter and apply it to the input

    Arguments:
    image (np.array): should be a np.uint16 array of the cell image
    random (bool): choose True to randomly assign the maximal intensity position
    sigma (tuple of integers): 5 is the recommended default, can adjust for different application

    Return:
    image*gaussian (np.array): cell image with gaussian pettern applied
    '''
    size = image.shape
    if not random:
        mean_x, mean_y = size[0]//2, size[1]//2
    else:
        x_coords, y_coords = np.nonzero(image)
        coordinates = list(zip(y_coords, x_coords))
        coord = coordinates[int(np.random.randint(len(coordinates)))]
        mean_x, mean_y = coord[0], coord[1]

    sigma_x, sigma_y = sigma[0], sigma[1]
    x_arr = np.arange(0, size[0])
    y_arr = np.arange(0, size[1])
    x_arr, y_arr = np.meshgrid(y_arr, x_arr)
    gaussian = np.exp(-((x_arr - mean_x)**2 / (2 * sigma_x**2) +\
                        (y_arr - mean_y)**2 / (2 * sigma_y**2)))
    return image*gaussian

if __name__ == '__main__':
    #cell = SynCell(50,50,45000,10)
    #cell2 = SynCell(150,170,30000,20)
    #io.imsave('test7.png', cell.cell_mask.astype(np.uint8)*255)
    #io.imsave('test8.png', cell.cell_img)
    canvas = SynCellImage(120,120)#2048,2048)
    #canvas.add_cells(cell)
    #canvas.add_cells(cell2)
    #io.imsave('test9.png', canvas.image)
    #io.imsave('test10.png', canvas.label)
    for _ in range(15):
        canvas.add_cells(SynCell((np.random.randint(10,110), np.random.randint(40,110)),\
                                 np.random.randint(20000,60000), np.random.randint(10,30),\
                                 np.random.uniform(1,2), np.random.randint(0,359)))
    plt.figure()
    plt.imshow(canvas.image)#, cmap='afmhot')
    plt.colorbar()
    plt.savefig('cellimage.png')

    plt.figure()
    plt.imshow(canvas.label, cmap='viridis')
    plt.colorbar()
    plt.savefig('celllabel.png')

    #canvas = SynCellImage(200,200)

    #for _ in range(1):
    #    canvas.add_cells(SynCell((np.random.randint(199,200), np.random.randint(199,200)),\
    #                              np.random.randint(30000,60000), np.random.randint(100,120),\
    #                              np.random.uniform(1,1.0001), np.random.randint(0,359)))
    io.imsave('cellimage_original.png', canvas.image)
    io.imsave('celllabel_original.png', canvas.label)
    #canvas.delete_cells(0)
    #io.imsave('test26.png', canvas.image)

#def draw_circle(center_x, center_y):