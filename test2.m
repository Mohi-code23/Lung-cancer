% Read in image and extract lungs
I = imread("E:\ENG Sem 6\Projects\Lung cancer\lung.jpg");
figure, imshow(I, []);
title('Original image');

preprocessedI = lungExtraction(I, strel('disk', 15), 1);

I_pre = pre2(preprocessedI);
figure, imshow(I_pre, []);
title('Preprocessed image')

% Complement the gabor image to change the background to white from black
I_pre = imcomplement(I_pre);
figure, imshow(I_pre, []);
title('Complement of Preprocessed image')

% Apply watershed on the preprocessed image
label = markcontrwatershed(I_pre, strel('disk', 10), strel(ones(5, 5)));

% Ground truth
I_gt = imread("E:\ENG Sem 6\Projects\Lung cancer\lung.jpg");
figure, imshow(I_gt, []);
title('Cancer location');

% Get suspicious nodules
nodules = noduleExtraction(label, preprocessedI);

% Label suspicious nodules
[L, num] = bwlabel(nodules);
disp(num);
RGB = label2rgb(L);
figure, imshow(RGB);
title('Detected suspicious nodules extracted');

% To choose a certain object
figure, imshow(L == 2);
title('Object 2');

figure,
imshow(I_pre, []);
hold on
himage = imshow(RGB);
himage

function [result] = gabor(image)
    lambda = 3;
    theta = 0;
    psi = [0 pi/2];
    gamma = 0.5;
    bw = 1.5;
    N = 16;
    img_in = im2double(image);
    img_out = zeros(size(img_in,1), size(img_in,2), N);
    
    for n = 1:N
        gb = gabor_fn(bw,gamma,psi(1),lambda,theta)...
            + 1i * gabor_fn(bw,gamma,psi(2),lambda,theta);
        img_out(:,:,n) = imfilter(img_in, gb, 'symmetric');
        theta = theta + 2*pi/N;
    end
    
    img_out_disp = sum(abs(img_out).^2, 3).^0.5;
    img_out_disp = img_out_disp./max(img_out_disp(:));
    result = img_out_disp;
end

function gb = gabor_fn(bw, gamma, psi, lambda, theta)
    sigma = lambda/pi*sqrt(log(2)/2)*(2^bw+1)/(2^bw-1);
    sigma_x = sigma;
    sigma_y = sigma/gamma;
    sz = fix(8*max(sigma_y, sigma_x));
    if mod(sz, 2) == 0
        sz = sz + 1;
    end
    
    [x, y] = meshgrid(-fix(sz/2):fix(sz/2), fix(sz/2):-1:fix(-sz/2));
    x_theta = x*cos(theta) + y*sin(theta);
    y_theta = -x*sin(theta) + y*cos(theta);
     
    gb = exp(-0.5*(x_theta.^2/sigma_x^2 + y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);
end

function [I_m_g_h_gab] = pre1(I)
    I_m = medfilt2(I, [3 3]);
    k = fspecial('gaussian', [5 5], 2);
    I_m_g = imfilter(I_m, k);
    I_m_g_h_gab = gabor(I_m_g);
end

function [I_gab] = pre2(I)
    I_gab = gabor(I);
end

function [I_m_g] = pre3(I)
    I_m = medfilt2(I, [3 3]);
    k = fspecial('gaussian', [5 5], 2);
    I_m_g = imfilter(I_m, k);
end

function [I_m] = pre4(I)
    I_m = medfilt2(I, [3 3]);
end

function [I_g] = pre5(I)
    k = fspecial('gaussian', [3 3], 2);
    I_g = imfilter(I, k);
end

function [I_h] = pre6(I)
    I_h = adapthisteq(I);
end

function [maskedImage] = lungExtraction(I, fill_strel, num_blobs)
    ThreshI = I < mean2(I);
    ThreshI = imclearborder(ThreshI);
    SE = fill_strel;
    ThreshI = imclose(ThreshI, SE);
    ThreshI = bwareafilt(ThreshI, num_blobs, 8);
    maskedImage = I;
    maskedImage(~ThreshI) = 0;
    imshow(maskedImage, []);
    title('Lung Extraction');
end

function [labels] = markcontrwatershed(I, openclosestrel, cleanstrel)
    Ie = imerode(I, openclosestrel);
    Iobr = imreconstruct(Ie, I);
    Iobrd = imdilate(Iobr, openclosestrel);
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
    Iobrcbr = imcomplement(Iobrcbr);
    fgm = imregionalmax(Iobrcbr, 8);
    I2 = I;
    I2(fgm) = 255;
    se2 = cleanstrel;
    Io = imopen(Iobrcbr, se2);
    Ie2 = imerode(Io, se2);
    Iobrcbr = imreconstruct(Ie2, Io);
    I3 = watershed(Iobrcbr);
    labels = I3 > 1;
end

function [nodules] = noduleExtraction(label, preprocessed_image)
    nodules = label;
    nodules(~preprocessed_image) = 0;
end

