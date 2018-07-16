function w_img = weight_image( w )
[Kh,Kw,Fi,Fo] = size(w);

Fi_img = Fi;
Fo_img = Fo;

if Fo == 1
   Fi_img = 1;
   Fo_img = Fi;
end

width = Kw * Fi_img;
height = Kh * Fo_img;

hb_inc = 0;
wb_inc = 0;

if Fo_img > 1
    hb_inc = 1;
    height = height + (Fo_img - 1);
end

if Fi_img > 1
    wb_inc = 1;
    width = width + (Fi_img - 1);
end

w_img = zeros(height,width);
mask = zeros(size(w_img));
by = 1:Kh;
bx = 1:Kw;

hb_out = 0;
wb_out = 0;

for fo = 1:Fo
    for fi = 1:Fi
        w_img(by,bx) = w(:,:,fi,fo);
        mask(by,bx) = 1;
        bx = bx + (Kw + wb_inc);
        
        if bx(1) > width
            hb_out = hb_out + 1;
            by = by + (Kh + hb_inc);            
            wb_out = 0;
            bx = 1:Kw;
        end
    end  
end

max_abs = max(abs(w(:)));
scale = 127 / max_abs;

w_img = scale * w_img;
w_img = w_img + 127;
w_img = double(w_img);
w_img(mask == 0) = 0;

colours = jet(256);
w_img = floor(w_img);

mask = mask == 0;
w_img_col = colours(w_img+1,:)';
w_img_colours(:,:,1) = reshape(w_img_col(1,:), height, width);
w_img_colours(:,:,2) = reshape(w_img_col(2,:), height, width);
w_img_colours(:,:,3) = reshape(w_img_col(3,:), height, width);
red = w_img_colours(:,:,1);
red(mask) = 0.5;
green = w_img_colours(:,:,2);
green(mask) = 0.5;
blue = w_img_colours(:,:,3);
blue(mask) = 0.5;

w_img_colours(:,:,1) = red;
w_img_colours(:,:,2) = green;
w_img_colours(:,:,3) = blue;
w_img = uint8(w_img_colours*255);

figure(5);
image(w_img); colormap(gray(256)); axis image; axis off;

end

