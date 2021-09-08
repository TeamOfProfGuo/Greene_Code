
import os

fpath = '../dataset/PLOT'

path_b = os.path.join(fpath, 'res18_baseline')
path_18 = os.path.join(fpath, 'res18')
path_50 = os.path.join(fpath, 'res50c')



import sys
from PIL import Image
for i in range(600):
    fname='0'*(3-len(str(i)))+str(i)


    images = [Image.open(x) for x in [os.path.join(path_b,fname,'rgb.jpg'),
                                      os.path.join(fpath, 'dep',fname+'.jpg'),
                                      os.path.join(path_b,fname,'gt.jpg'),
                                      os.path.join(path_b,fname,'pred_path.jpg'),
                                      os.path.join(path_18,fname,'pred_path.jpg'),
                                      os.path.join(path_50,fname,'pred_path.jpg'),
                                      ]]
    widths, heights = zip(*(i.size for i in images))

    total_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (total_width, total_height))

    y_offset = 0
    for im in images:
      new_im.paste(im, (0, y_offset))
      y_offset += im.size[1]

    new_im.save(os.path.join(fpath, 'all', fname+'.jpg'))
    print('finish saving image'+fname)
