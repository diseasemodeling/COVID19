import sys
from PIL import Image
def merge_image():
  ##combine 2, 3, 4
  #img1 = Image.open('1.png')
  images = [Image.open(x) for x in ['8.png','1.png']]#,'4.png','5.png']]#,'6.png']]#,'plot7.jpg','plot3.jpg','plot4.jpg']]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test1.png')

  images = [Image.open(x) for x in ['2.png','3.png']]#,'4.png','5.png']]#,'6.png']]#,'plot7.jpg','plot3.jpg','plot4.jpg']]
  widths, heights = zip(*(i.size for i in images))

  total_width = sum(widths)
  max_height = max(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (x_offset,0))
    x_offset += im.size[0]

  new_im.save('test2.png')

  images = [Image.open(x) for x in ['test1.png','test2.png']]#,'4.png','5.png']]#,'6.png']]#,'plot7.jpg','plot3.jpg','plot4.jpg']]
  widths, heights = zip(*(i.size for i in images))

  total_width = max(widths)
  max_height = sum(heights)

  new_im = Image.new('RGB', (total_width, max_height))

  x_offset = 0
  for im in images:
    new_im.paste(im, (0,x_offset))
    x_offset += im.size[1]

  new_im.save('results.png')
