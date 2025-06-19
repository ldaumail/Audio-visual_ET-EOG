function [pix] = angle2pix(display, ang)
    pixSize = display.screenWidth / display.resolution(1); 
    sz = 2 * display.viewingDist * tan(pi * ang / (2 * 180)); 
    pix = round(sz / pixSize); 
end