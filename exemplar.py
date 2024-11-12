from main import *

canvas = SynCellImage(120,120)
for _ in range(10):
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


io.imsave('cellimage_original.png', canvas.image)
io.imsave('celllabel_original.png', canvas.label)

canvas.delete_cells(2)
canvas.describe()
canvas.cell_dict[0].describe()