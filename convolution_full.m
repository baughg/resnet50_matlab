function layer1 = convolution_full( input_image, weight0_1, bias0_1, pad )
[H,W,C] = size(input_image);
[Kh,Kw,Fo,Fi] = size(weight0_1);


row_pad = (Kh - 1) / 2;
col_pad = (Kw - 1) / 2;

layer1 = zeros(H,W,Fo);

if(pad == 0)
    layer1 = zeros(H-2*row_pad,W-2*col_pad,Fo);
    row_pad = 0;
    col_pad = 0;
end



input_image_pad = zeros(H+2*row_pad,W+2*col_pad);

for c = 1:C
    input_image_pad((row_pad+1):(end-row_pad),(col_pad+1):(end-col_pad)) = input_image(:,:,c);
    
    for f = 1:Fo
        kernel = weight0_1(:,:,f,c);
        kernel = rot90(squeeze(kernel),2);
        conv_out = conv2(input_image_pad,kernel,'valid');
        layer1(:,:,f) =  layer1(:,:,f) + conv_out;      
    end
end

for f = 1:Fo
    layer1(:,:,f) = layer1(:,:,f) + bias0_1(f);
%     layer1(:,:,f) = layer1(:,:,f) .* double(layer1(:,:,f) > 0); % relu
end

layer1 = squeeze(layer1);
end

