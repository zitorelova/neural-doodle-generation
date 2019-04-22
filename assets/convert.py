from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM

draw = svg2rlg('sample.svg')
renderPM.drawToFile(draw, 'file_test.jpg', fmt='JPG')

